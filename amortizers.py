import bayesflow.default_settings as defaults
import numpy as np
import tensorflow as tf
from bayesflow.amortizers import AmortizedPosterior
from skimage.util import random_noise
from scipy.ndimage import gaussian_filter

# Training helpers
from bayesflow.exceptions import SummaryStatsError
from bayesflow.losses import mmd_summary_space
from tensorflow.keras import regularizers
from tensorflow.keras import layers, Input, Model


@tf.function
def discretize_time(eps, T_max, num_steps, rho=7.0):
    """Function for obtaining the discretized time according to
    https://arxiv.org/pdf/2310.14189.pdf, Section 2, bottom of page 2.

    Parameters:
    -----------
    T_max   : int
        Maximal time (corresponds to $\sigma_{max}$)
    eps     : float
        Minimal time (correspond to $\sigma_{min}$)
    N       : int
        Number of discretization steps
    rho     : number
        Control parameter
    """
    N = tf.cast(num_steps, tf.float32) + 1.0
    i = tf.range(1, N + 1, dtype=tf.float32)
    one_over_rho = 1.0 / rho
    discretized_time = (
        eps**one_over_rho + (i - 1.0) / (N - 1.0) * (T_max**one_over_rho - eps**one_over_rho)
    ) ** rho
    return discretized_time


class ConfigurableHiddenBlock(tf.keras.Model):
    def __init__(
        self, num_units, activation="relu", residual_connection=True, dropout_rate=0.0, kernel_regularization=0.0
    ):
        super().__init__()

        self.act_func = tf.keras.activations.get(activation)
        self.residual_connection = residual_connection
        self.dense = tf.keras.layers.Dense(
            num_units, activation=None, kernel_regularizer=regularizers.l2(kernel_regularization)
        )
        self.dropout_rate = dropout_rate

    @tf.function
    def call(self, inputs, training=False, mask=None):
        x = self.dense(inputs)
        x = tf.nn.dropout(x, self.dropout_rate)

        if self.residual_connection:
            x += inputs
        return self.act_func(x)


# Source code for networks adapted from: https://keras.io/examples/generative/ddpm/
# Kernel initializer to use
def kernel_init(scale):
    scale = max(scale, 1e-10)
    return tf.keras.initializers.VarianceScaling(scale, mode="fan_avg", distribution="uniform")

# class TimeEmbedding(tf.keras.layers.Layer):
#     def __init__(self, dim, tmax, **kwargs):
#         super().__init__(**kwargs)
#         self.dim = dim
#         self.tmax = tmax
#         self.half_dim = dim // 2
#         self.emb = tf.math.log(10000.0) / (self.half_dim - 1)
#         self.emb = tf.exp(tf.range(self.half_dim, dtype=tf.float32) * -self.emb)

#     @tf.function
#     def call(self, inputs):
#         inputs = tf.cast(inputs, dtype=tf.float32) * 1000.0 / self.tmax
#         emb = inputs[:, None] * self.emb[None, :]
#         emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
#         return emb

class TimeEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim, tmax, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.tmax = tmax
        self.half_dim = dim // 2
        self.freqs = tf.exp(
        -tf.math.log(self.tmax) * tf.range(0, self.half_dim, dtype=tf.float32) / self.half_dim
    )

    @tf.function
    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        inputs = inputs[:, tf.newaxis] * self.freqs[tf.newaxis]
        embedding = tf.concat([tf.cos(inputs), tf.sin(inputs)], axis=-1)
        if self.dim % 2:
            embedding = tf.concat([embedding, tf.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

def TimeMLP(units, activation_fn=tf.keras.activations.swish):
    def apply(inputs):
        temb = tf.keras.layers.Dense(units, activation=activation_fn, kernel_initializer=kernel_init(1.0))(inputs)
        temb = tf.keras.layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        return temb

    return apply

class ConfigurableMLP(tf.keras.Model):
    """Implements a configurable MLP with optional residual connections and dropout."""

    def __init__(
        self,
        input_dim,
        condition_dim,
        hidden_dim=512,
        num_hidden=2,
        activation="relu",
        residual_connections=True,
        dropout_rate=0.0,
        kernel_regularization=0.0,
    ):
        """
        Creates an instance of a flexible MLP with optional residual connections
        and dropout.

        Parameters:
        -----------
        input_dim : int
            The input dimensionality
        condition_dim  : int
            The dimensionality of the condition
        hidden_dim: int, optional, default: 512
            The dimensionality of the hidden layers
        num_hidden: int, optional, default: 2
            The number of hidden layers (minimum 1)
        eps       : float, optional, default: 0.002
            The minimum time
        activation: string, optional, default: 'relu'
            The activation function of the dense layers
        T_max     : float, optional, default: 0.20
            End time of the diffusion
        N         : int, optional, default: s1
            Discretization level during inference
        residual_connections: bool, optional, default: True
            Use residual connections in the MLP
        dropout_rate        : float, optional, default: 0.0
            Dropout rate for the hidden layers in the MLP
        kernel_regularization: float, optional, default: 0.0
            L2 regularization factor for the kernel weights
        """
        # super(ConfigurableMLP, self).__init__()
        super().__init__()

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = input_dim  # only for compatibility with bayesflow.amortizers.AmortizedPosterior

        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    hidden_dim, activation=activation, kernel_regularizer=regularizers.l2(kernel_regularization)
                ),
            ]
        )
        for _ in range(num_hidden):
            self.model.add(
                ConfigurableHiddenBlock(
                    hidden_dim,
                    activation=activation,
                    residual_connection=residual_connections,
                    dropout_rate=dropout_rate,
                    kernel_regularization=kernel_regularization,
                )
            )
        self.model.add(tf.keras.layers.Dense(input_dim))

    @tf.function
    def call(self, inputs, training=False, mask=None):
        return self.model(tf.concat(inputs, axis=-1), training=training)



def build_mlp(input_dim,
        condition_dim,
        hidden_dim=512,
        use_time_embedding=False,
        T_max=200.0,
        num_hidden=2,
        activation="relu",
        residual_connections=True,
        dropout_rate=0.0,
        kernel_regularization=0.0,):
    
    use_time_embedding = use_time_embedding


    x_input = Input(shape=(input_dim), name="x_input")
    
    time_input = Input(shape=(), dtype=tf.float32, name="time_input")
    condition_input = Input(shape=(condition_dim), dtype=tf.float32, name="condition_input")

    t = time_input
    if use_time_embedding:
        t = TimeEmbedding(dim=32, tmax=T_max)(time_input)
        t = TimeMLP(units=32, activation_fn="relu")(t)
    else:
        t = t[..., tf.newaxis]
    
    x = layers.Concatenate(axis=-1)([x_input, condition_input, t])

    x = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                hidden_dim, activation=activation, kernel_regularizer=regularizers.l2(kernel_regularization)
            ),
        ]
    )(x)

    for _ in range(num_hidden):
        x = ConfigurableHiddenBlock(
                hidden_dim,
                activation=activation,
                residual_connection=residual_connections,
                dropout_rate=dropout_rate,
                kernel_regularization=kernel_regularization,
            )(x)
    x = tf.keras.layers.Dense(input_dim)(x)

    model = Model([x_input, condition_input, time_input], x, name="MLP")
    model.input_dim = input_dim
    model.condition_dim = condition_dim
    model.latent_dim = input_dim  # only for compatibility with bayesflow.amortizers.AmortizedPosterior

    return model


class ConsistencyAmortizer(AmortizedPosterior):
    """Implements a consistency model according to https://arxiv.org/abs/2303.01469"""

    def __init__(
        self,
        consistency_net,
        num_steps,
        summary_net=None,
        loss_fun=None,
        summary_loss_fun=None,
        sigma2=1.0,
        eps=0.001,
        T_max=200.0,
        s0=10,
        s1=50,
        **kwargs,
    ):
        """
        Creates an instance of a consistency model (CM) to be used
        for standalone consistency training (CT).

        Parameters:
        -----------
        consistency_net         : tf.keras.Model
            A neural network for the consistency model
        input_dim : int
            The input dimensionality
        condition_dim  : int
            The dimensionality of the condition (or summary net output)
        num_steps: int
            The total number of training steps
        summary_net       : tf.keras.Model or None, optional, default: None
            An optional summary network to compress non-vector data structures.
        loss_fun          : callable or None, optional, default: None
            TODO: Currently unused, remove or implement, add documentation
        summary_loss_fun  : callable, str, or None, optional, default: None
            The loss function which accepts the outputs of the summary network. If ``None``, no loss is provided
            and the summary space will not be shaped according to a known distribution (see [2]).
            If ``summary_loss_fun='MMD'``, the default loss from [2] will be used.
        sigma2      : np.ndarray of shape (input_dim, 1), or float, optional, default: 1.0
            Controls the shape of the skip-function
        eps         : float, optional, default: 0.001
            The minimum time
        T_max       : flat, optional, default: 200.0
            The end time of the diffusion
        s0          : int, optional, default: 10
            Initial discretization steps
        s1          : int, optional, default: 50
            Final discretization steps
        **kwargs          : dict, optional, default: {}
            Additional keyword arguments passed to the ``__init__`` method of a ``tf.keras.Model`` instance via AmortizedPosterior.

        Important
        ----------
        - If no ``summary_net`` is provided, then the output dictionary of your generative model should not contain
        any ``summary_conditions``, i.e., ``summary_conditions`` should be set to ``None``, otherwise these will be ignored.
        """

        super().__init__(consistency_net, **kwargs)

        self.input_dim = consistency_net.input_dim
        self.condition_dim = consistency_net.condition_dim

        self.img_size = 32

        self.student = consistency_net
        self.student.build(
            input_shape=(
                None,
                self.input_dim + self.condition_dim + 1,
            )
        )

        self.summary_net = summary_net
        if loss_fun is not None:
            raise NotImplementedError("Only the default pseudo-huber loss is currently supported.")
        # self.loss_fun = self._determine_loss(loss_fun)
        self.summary_loss = self._determine_summary_loss(summary_loss_fun)

        self.sigma2 = tf.Variable(sigma2)
        self.sigma = tf.Variable(tf.math.sqrt(sigma2))
        self.eps = eps
        self.T_max = T_max
        # Choose coefficient according to https://arxiv.org/pdf/2310.14189.pdf, Section 3.3
        self.c_huber = 0.00054 * tf.sqrt(tf.cast(self.input_dim, tf.float32))
        self.c_huber2 = tf.square(self.c_huber)

        self.num_steps = tf.cast(num_steps, tf.float32)
        self.s0 = tf.cast(s0, tf.float32)
        self.s1 = tf.cast(s1, tf.float32)

        self.current_step = tf.Variable(0, trainable=False, dtype=tf.float32)

    @tf.function
    def call(self, input_dict, z, t, return_summary=False, **kwargs):
        """Performs a forward pass through the summary and consistency network given an input dictionary.

        Parameters
        ----------
        input_dict     : dict
            Input dictionary containing the following mandatory keys, if ``DEFAULT_KEYS`` unchanged:
            ``targets``            - the latent model parameters over which a condition density is learned
            ``summary_conditions`` - the conditioning variables (including data) that are first passed through a summary network
            ``direct_conditions``  - the conditioning variables that the directly passed to the inference network
        z              : tf.Tensor of shape (batch_size, input_dim)
            The noise vector
        t              : tf.Tensor of shape (batch_size, 1)
            Vector of time samples in [eps, T]
        return_summary : bool, optional, default: False
            A flag which determines whether the learnable data summaries (representations) are returned or not.
        **kwargs       : dict, optional, default: {}
            Additional keyword arguments passed to the networks
            For instance, ``kwargs={'training': True}`` is passed automatically during training.

        Returns
        -------
        net_out or (net_out, summary_out)
        """
        # Concatenate conditions, if given
        summary_out, full_cond = self._compute_summary_condition(
            input_dict.get(defaults.DEFAULT_KEYS["summary_conditions"]),
            input_dict.get(defaults.DEFAULT_KEYS["direct_conditions"]),
            **kwargs,
        )
        # Extract target variables
        target_vars = input_dict[defaults.DEFAULT_KEYS["parameters"]]

        # Compute output
        inp = target_vars + t[:, None, None] * z
        net_out = self.consistency_function(inp, full_cond, t, **kwargs)

        # Return summary outputs or not, depending on parameter
        if return_summary:
            return net_out, summary_out
        return net_out

    @tf.function
    def consistency_function(self, x, c, t, **kwargs):
        """Compute consistency function.

        Parameters
        ----------
        x : tf.Tensor of shape (batch_size, input_dim)
            Input vector
        c : tf.Tensor of shape (batch_size, condition_dim)
            The conditioning vector
        t : tf.Tensor of shape (batch_size, 1)
            Vector of time samples in [eps, T]
        """
        F = self.student([x, c, t], **kwargs)

        # if len(F.shape) == 4:
        #   batch_size = tf.shape(F)[0]
        #   flattened_dim = tf.reduce_prod(tf.shape(F)[1:])
        #   F = tf.reshape(F, (batch_size, flattened_dim))

        # Compute skip and out parts (vectorized, since self.sigma2 is of shape (1, input_dim)
        # Thus, we can do a cross product with the time vector which is (batch_size, 1) for
        # a resulting shape of cskip and cout of (batch_size, input_dim)
        cskip = self.sigma2 / ((t - self.eps) ** 2 + self.sigma2)
        cout = self.sigma * (t - self.eps) / (tf.math.sqrt(self.sigma2 + t**2))

        out = cskip[:, None, None] * x + cout[:, None, None] * F
        return out

    def compute_loss(self, input_dict, **kwargs):
        """Computes the loss of the posterior amortizer given an input dictionary, which will
        typically be the output of a Bayesian ``GenerativeModel`` instance.

        Parameters
        ----------
        input_dict : dict
            Input dictionary containing the following mandatory keys, if ``DEFAULT_KEYS`` unchanged:
            ``targets``            - the latent variables over which a condition density is learned
            ``summary_conditions`` - the conditioning variables that are first passed through a summary network
            ``direct_conditions``  - the conditioning variables that the directly passed to the inference network
        z          : tf.Tensor of shape (batch_size, input_dim)
            The noise vector
        t1         : tf.Tensor of shape (batch_size, 1)
            Vector of time samples in [eps, T]
        t2         : tf.Tensor of shape (batch_size, 1)
            Vector of time samples in [eps, T]
        TODO: add documentation for c, t1, t2
        **kwargs   : dict, optional, default: {}
            Additional keyword arguments passed to the networks
            For instance, ``kwargs={'training': True}`` is passed automatically during training.

        Returns
        -------
        total_loss : tf.Tensor of shape (1,) - the total computed loss given input variables
        """
        self.current_step.assign_add(1.0)

        # Extract target variables and generate noise
        theta = input_dict.get(defaults.DEFAULT_KEYS["parameters"])
        z = tf.random.normal(tf.shape(theta))

        N_current = self._schedule_discretization(self.current_step, self.num_steps, s0=self.s0, s1=self.s1)
        discretized_time = discretize_time(self.eps, self.T_max, N_current)

        # Randomly sample t_n and t_[n+1] and reshape to (batch_size, 1)
        # adapted noise schedule from https://arxiv.org/pdf/2310.14189.pdf,
        # Section 3.5
        P_mean = -1.1
        P_std = 2.0
        log_p = tf.math.log(
            tf.math.erf((tf.math.log(discretized_time[1:]) - P_mean) / (tf.sqrt(2.0) * P_std))
            - tf.math.erf((tf.math.log(discretized_time[:-1]) - P_mean) / (tf.sqrt(2.0) * P_std))
        )
        times = tf.random.categorical([log_p], tf.shape(theta)[0])[0]
        t1 = tf.gather(discretized_time, times)[..., None]
        t2 = tf.gather(discretized_time, times + 1)[..., None]

        # Teacher is just the student without gradient tracing
        teacher_out = tf.stop_gradient(self(input_dict, z, t1, return_summary=False, **kwargs))
        student_out, sum_out = self(input_dict, z, t2, return_summary=True, **kwargs)
        # weighting function, see https://arxiv.org/pdf/2310.14189.pdf, Section 3.1
        lam = 1 / (t2 - t1)
        # Pseudo-huber loss, see https://arxiv.org/pdf/2310.14189.pdf, Section 3.3
        loss = tf.reduce_mean(lam[:, None, None] * (tf.sqrt(tf.square(teacher_out - student_out) + self.c_huber2) - self.c_huber))

        # Case summary loss should be computed
        if self.summary_loss is not None:
            sum_loss = self.summary_loss(sum_out)
        # Case no summary loss, simply add 0 for convenience
        else:
            sum_loss = 0.0

        # Compute and return total loss
        total_loss = tf.reduce_mean(loss) + sum_loss
        return total_loss


    def deblur_gaussian_wiener(self, blurred_tensor, psf_sigma=2.5, K=1e-3):
          """
          Deblurs a batch of blurred Fashion MNIST images using Wiener deconvolution.
          
          Parameters:
          ----------
          blurred_tensor : tf.Tensor of shape [batch_size, 784]
              Flattened, blurred grayscale images (28x28).
          psf_sigma : float
              Standard deviation of Gaussian PSF used in the blur.
          K : float
              Regularization strength (higher = more smoothing, less noise amplification).
              
          Returns:
          -------
          deblurred_tensor : tf.Tensor of shape [batch_size, 784]
              Deblurred image tensor (flattened 28x28).
          """
          size = self.img_size
          batch_size = tf.shape(blurred_tensor)[0]

          # Generate Gaussian PSF
          ax = np.arange(-size // 2 + 1., size // 2 + 1.)
          xx, yy = np.meshgrid(ax, ax)
          psf = np.exp(-(xx**2 + yy**2) / (2. * psf_sigma**2))
          psf /= np.sum(psf)

          # Center PSF in a 28x28 grid
          psf_padded = np.zeros((size, size))
          cx = size // 2 - psf.shape[0] // 2
          cy = size // 2 - psf.shape[1] // 2
          psf_padded[cx:cx+psf.shape[0], cy:cy+psf.shape[1]] = psf

          # Get Wiener filter in Fourier domain
          H_f = np.fft.fft2(np.fft.ifftshift(psf_padded))
          H_f_conj = np.conj(H_f)
          wiener_filter = H_f_conj / (np.abs(H_f)**2 + K)
          wiener_filter_tf = tf.constant(wiener_filter, dtype=tf.complex64)

          # Apply FFT-based deconvolution
          blurred_imgs = tf.reshape(blurred_tensor, [-1, size, size])
          blurred_fft = tf.signal.fft2d(tf.cast(blurred_imgs, tf.complex64))
          deblurred_fft = blurred_fft * wiener_filter_tf
          deblurred_imgs = tf.math.real(tf.signal.ifft2d(deblurred_fft))
          return tf.reshape(deblurred_imgs, [-1, size*size])


    def sample_implicit(self, input_dict, n_samples, n_steps=10, to_numpy=True, step_size=1e-3, theta=50, **kwargs):
        """Generates random draws from the approximate posterior given a dictionary with conditonal variables
        using the multistep sampling algorithm (Algorithm 1).

        Parameters
        ----------
        input_dict  : dict
            Input dictionary containing the following mandatory keys, if ``DEFAULT_KEYS`` unchanged:
            ``summary_conditions`` : the conditioning variables (including data) that are first passed through a summary network
            ``direct_conditions``  : the conditioning variables that the directly passed to the inference network
        n_samples   : int
            The number of posterior draws (samples) to obtain from the approximate posterior
        n_steps     : int
            The number of sampling steps
        TODO: This does not seem to work in some cases
        to_numpy    : bool, optional, default: True
            Flag indicating whether to return the samples as a ``np.ndarray`` or a ``tf.Tensor``
        **kwargs    : dict, optional, default: {}
            Additional keyword arguments passed to the networks

        Returns
        -------
        post_samples : tf.Tensor or np.ndarray of shape (n_data_sets, n_samples, n_params)
            The sampled parameters from the approximate posterior of each data set
        """

        # Compute condition (direct, summary, or both)
        _, conditions = self._compute_summary_condition(
            input_dict.get(defaults.DEFAULT_KEYS["summary_conditions"]),
            input_dict.get(defaults.DEFAULT_KEYS["direct_conditions"]),
            training=False,
            **kwargs,
        )
        n_data_sets, condition_dim = tf.shape(conditions)

        #conds = input_dict.get(defaults.DEFAULT_KEYS["summary_conditions"])
        #conds = self.deblur_gaussian_wiener(conds)

        assert condition_dim == self.condition_dim

        post_samples = np.empty(shape=(n_data_sets, n_samples, self.input_dim), dtype=np.float32)
        n_data_sets, condition_dim = conditions.shape

        for i in range(n_data_sets):
            c = conditions[i, None]
            c_rep = tf.concat([c] * n_samples, axis=0)
            discretized_time = tf.reverse(discretize_time(self.eps, self.T_max, n_steps), axis=[-1])
            z_init = tf.random.normal((n_samples, self.input_dim), stddev=self.T_max)
            T = discretized_time[0] + tf.zeros((n_samples, 1))
            x_n = z_init
            #t_n = discretized_time[0] 
            #t_nm1 = discretized_time[1]

            #sigma_nm1 = tf.math.sqrt( (t_nm1**2 * (t_n**2 - t_nm1**2))/(t_n**2) )

            #c1 = tf.math.sqrt( (t_nm1**2 )/(t_n**2) )
            #c2 = 1 - c1

            #x_n = conds[i, None]
            #x_n = tf.concat([x_n] * n_samples, axis=0) + 0.1*z_init

            
            samples = self.consistency_function(x_n, c_rep, T)

            eta = 0
            #teta = 50
            theta = tf.cast(theta, tf.float32)
        
            for n in range(1, n_steps):
                z = tf.random.normal((n_samples, self.input_dim))
                # alpha_nm1 = 1 /  (discretized_time[n-1] ** 2 - self.eps**2 + 1)
                # alpha_n = 1 /  (discretized_time[n] ** 2 - self.eps**2 + 1)

                t_n = discretized_time[n] 
                t_nm1 = discretized_time[n+1]

                sigma_nm1 = eta * tf.math.sqrt( (t_nm1**2 * (t_n**2 - t_nm1**2))/(t_n**2) )
                
                s1 =  (theta  )*(t_nm1**2 - sigma_nm1**2)/(t_n**2)
                s2 = (1  )*(t_nm1**2)/(t_n**2)
                c1 = tf.math.sqrt( s1 )
                c2 = tf.math.sqrt( 1 - s2 )
                
                # c1 = 50*tf.math.sqrt( s1 )
                # c2 = tf.math.sqrt( 1 - s2 )

                # c1 = tf.math.sqrt( s1 )
                # c2 = 1 - tf.math.sqrt(s2 )

                x_nm1 = c1*x_n + c2*samples + (sigma_nm1) * z

                #x_nm1 = x_nm1 - 0.1 * ((self.deblur_gaussian(c_rep) - samples)**2)

                samples = self.consistency_function(x_nm1, c_rep, t_n + tf.zeros((n_samples, 1)))
                x_n = x_nm1
                
            post_samples[i] = samples

        # Remove trailing first dimension in the single data case
        if n_data_sets == 1:
            post_samples = tf.squeeze(post_samples, axis=0)

        # Return numpy version of tensor or tensor itself
        if to_numpy:
            return post_samples.numpy()
        return post_samples

    
        
    def inpainting_mask(self, images, mask_size=8):
      """
      Applies an inpainting mask to a batch of flattened images (shape: [batch_size, 784]).

      Parameters:
      -----------
      images     : tf.Tensor of shape (batch_size, 784)
      mask_size  : size of the square mask to apply on each image
      """
      batch_size = tf.shape(images)[0]
      height = width = self.img_size

      # Reshape to (batch_size, 28, 28)
      images_reshaped = tf.reshape(images, (batch_size, height, width))

      def mask_single_image(image):
          # Random top-left corner
          # top = tf.random.uniform([], 0, height - mask_size, dtype=tf.int32)
          # left = tf.random.uniform([], 0, width - mask_size, dtype=tf.int32)
          
          top = 10
          left = 10

          # Create a mask of ones
          mask = tf.ones_like(image)

          # Apply 0s in square region
          mask = tf.tensor_scatter_nd_update(
              mask,
              indices=tf.reshape(tf.stack(tf.meshgrid(tf.range(top, top + mask_size),
                                                      tf.range(left, left + mask_size),
                                                      indexing='ij'), axis=-1), [-1, 2]),
              updates=tf.zeros([mask_size * mask_size], dtype=image.dtype) - 1
          )

          return image * mask

      # Apply the mask to each image in the batch
      masked_images = tf.map_fn(mask_single_image, images_reshaped)

      # Flatten back to (batch_size, 784)
      return tf.reshape(masked_images, (batch_size, height * width))

    

    def blur(self, images):
        batch_size = tf.shape(images)[0]
        height = width = self.img_size

        images_reshaped = tf.reshape(images, (batch_size, height, width, 3))

        def grayscale_camera_np(image, noise="poisson", psf_width=2.5, noise_scale=1, noise_gain=0.5):
            image = noise_scale * image
            image = noise_gain * random_noise(image, mode=noise)
            image = np.stack([gaussian_filter(image[..., c], sigma=psf_width) for c in range(3)], axis=-1)
            return image.astype(np.float32)

        def tf_wrapper(image):
            return tf.numpy_function(
                func=grayscale_camera_np,
                inp=[image],
                Tout=tf.float32
            )

        masked_images = tf.map_fn(tf_wrapper, images_reshaped)

        return tf.reshape(masked_images, (batch_size, self.img_size, self.img_size, 3))
        
    

    def sample_addim(self,
           input_dict,
           n_samples,
           n_steps: int = 2,
           theta = 0.9,
           eta: float = 0.0,           # ← new hyperparameter
           to_numpy: bool = True,
           c1: float = 1.0,
           c2: float = 1.0,  
           img_size: int = 32,          
           **kwargs):
        """
        DDIM / consistency‐model sampler following eq. (9) in your notes:

            x_{t_{n-1}} = 
              sqrt((t_{n-1}^2 - σ_{n-1}^2) / t_n^2) · x_{t_n}
            + (1 - sqrt((t_{n-1}^2 - σ_{n-1}^2) / t_n^2)) · x₀_pred
            + σ_{n-1} · z,       z∼N(0,I)

        where σ_{n-1} = η · sqrt((t_n^2 − t_{n-1}^2)·t_{n-1}^2 / t_n^2).
        """

        self.img_size = img_size       

        # 1) compute conditioning
        _, conditions = self._compute_summary_condition(
            input_dict.get(defaults.DEFAULT_KEYS["summary_conditions"]),
            input_dict.get(defaults.DEFAULT_KEYS["direct_conditions"]),
            training=False,
            **kwargs
        )

        conds = input_dict.get(defaults.DEFAULT_KEYS["summary_conditions"])
        conds_d = self.deblur_gaussian_wiener(conds)
        conds_0 = conds
        
        n_data, _ = tf.shape(conditions)
        # 2) build the time‐grid t_N > … > t_0
        ts = tf.reverse(discretize_time(self.eps, self.T_max, n_steps), axis=[-1])
        # ts[n] = t_n.  ts[0] = T_max, ts[-1] = eps

        d = float(self.input_dim)

        out = []
        for i in range(n_data):
            c = conditions[i:i+1]                     # (1,cond_dim)
            #cond = conds_d                 # (1,cond_dim)
        
            c_rep = tf.repeat(c, n_samples, axis=0)   # (n_samples, cond_dim)
            cond_rep = tf.repeat(conds_d, n_samples, axis=0)   # (n_samples, input_dim)
            cond_rep0 = tf.repeat(conds_0, n_samples, axis=0)
            #print(cond_rep.shape) 
            # 3) sample initial x_{t_N} ∼ Normal(0, t_N^2 I)
            t_N = ts[0]
            x = tf.random.normal((n_samples, self.img_size, self.img_size, 3)) #* t_N

            # 4) DDIM loop
            for n in range(len(ts)-1):
                t_n     = ts[n]     # current time
                t_prev  = ts[n+1]   # next (smaller) time

                t_p = 1.0*ts[n]

                # ← predict clean x₀:
                x0_pred = self.consistency_function(
                    x, c_rep, 
                    tf.fill((n_samples,1), t_n)
                )
                
                s = tf.random.normal((n_samples, self.img_size, self.img_size, 3))
                x = x0_pred + t_p * s

                #print(x0_pred.shape)

                # calculate x_var

                #x_var1 = tf.reduce_sum((tf.reshape(cond_rep, (n_samples, 3*32*32)) - x0_pred)**2, axis=1, keepdims=True) / d
                # x_var = tf.norm((tf.reshape(cond_rep, (n_samples, 784)) - x0_pred), ord=2)**2
                # cc = -1.0 + 2 * tf.reshape(cond_rep0, (n_samples, 784)) / 255.0
                # x_var_0 = tf.norm((cc - self.inpainting_mask(tf.reshape(x0_pred, (n_samples,28,28)))), ord=2)**2 
                # x_var_1 = tf.norm((cc - x0_pred), ord=2)**2 
                # x_var_2 = 0.1 /(2 + t_p**2)
                

                cc = tf.reshape(cond_rep0, (n_samples, self.img_size, self.img_size, 3))
                x0_scaled = (np.clip(x0_pred, a_min=-1.00, a_max=1.00))
                cc = (1.00 + cc) / 2.00
                x_var_0 = tf.norm((cc - (1.00 + self.blur(x0_scaled))/2.00 ), ord=2)**2 
                x_var_1 = tf.norm((cc - x0_pred), ord=2)**2
                x_var_2 = tf.norm((tf.reshape(cond_rep, (n_samples, self.img_size, self.img_size, 3)) - x0_pred), ord=2)**2


                #print(f"x_var: {x_var}")
                # ← compute σ_{n-1} and the “α” coefficient a = sqrt((t_prev² − σ²)/t_n²)
                sigma = eta * tf.sqrt(
                    (t_p**2 - t_prev**2) * (t_prev**2) / (t_p**2)
                )
                a = tf.sqrt((t_prev**2 - sigma**2) / (t_p**2))
                #print(f"a: {a}")
                # ← the DDIM mean:
                err = (x - x0_pred)
                
                norm22 = tf.reduce_sum(err**2, axis=1, keepdims=True)
                norm2 = tf.norm(err,ord=2)**2
                if n == len(ts)-2:
                  err_coef = 0
                else:
                  # print(a**2)
                  # print(tf.reduce_min(x0_pred).numpy())
                  # print(tf.reduce_max(x0_pred).numpy())
                  # print(tf.reduce_min(self.blur(x0_scaled)).numpy())
                  # print(tf.reduce_max(self.blur(x0_scaled)).numpy())
                  # print(tf.reduce_min(cc).numpy())
                  # print(tf.reduce_max(cc).numpy())
                  # print(100*x_var_0*((1-a)**2)/norm2)
                  #print(x_var_0)
                  # print("")
                  err_coef = c1 * tf.sqrt(a**2 + c2*x_var_0*((1-a)**2)/norm2)#*((1-a)**2)/(norm2))#*((1.0 - a)**2)/norm2) 
                #err_coef = 5.90*tf.sqrt(a**2 + 1.0*x_var*((a))/norm2)#*((1-a)**2)/(norm2))#*((1.0 - a)**2)/norm2) 
                #err_coef = a
                
                #print(f"err_coef2: {err_coef2}")
                #print(f"err_coef: {err_coef}")
                #print(f"ration: {err_coef2/err_coef}")

                #print(f"a: {a}")

                x_mean = x0_pred + err_coef * err
                #x_mean = a * x + (1.0 - a) * x0_pred

                # ← add noise (if η>0)
                if eta > 0:
                    z = tf.random.normal((n_samples, self.img_size, self.img_size, 3))
                    x = x_mean + sigma * z
                else:
                    x = x_mean

            out.append(x)

        # stack and possibly squeeze
        post = tf.stack(out, axis=0)  # (n_data, n_samples, input_dim)
        if n_data == 1:
            post = tf.squeeze(post, 0)

        return post.numpy() if to_numpy else post
        

    def sample_addim2(self,
           input_dict,
           n_samples,
           n_steps: int = 10,
           eta: float = 0.99,           # ← new hyperparameter
           to_numpy: bool = True,
           **kwargs):
        """
        DDIM / consistency‐model sampler following eq. (9) in your notes:

            x_{t_{n-1}} = 
              sqrt((t_{n-1}^2 - σ_{n-1}^2) / t_n^2) · x_{t_n}
            + (1 - sqrt((t_{n-1}^2 - σ_{n-1}^2) / t_n^2)) · x₀_pred
            + σ_{n-1} · z,       z∼N(0,I)

        where σ_{n-1} = η · sqrt((t_n^2 − t_{n-1}^2)·t_{n-1}^2 / t_n^2).
        """

        # 1) compute conditioning
        _, conditions = self._compute_summary_condition(
            input_dict.get(defaults.DEFAULT_KEYS["summary_conditions"]),
            input_dict.get(defaults.DEFAULT_KEYS["direct_conditions"]),
            training=False,
            **kwargs
        )
        n_data, _ = tf.shape(conditions)

        # 2) build the time‐grid t_N > … > t_0
        ts = tf.reverse(discretize_time(self.eps, self.T_max, n_steps), axis=[-1])
        # ts[n] = t_n.  ts[0] = T_max, ts[-1] = eps

        out = []
        for i in range(n_data):
            c = conditions[i:i+1]                     # (1,cond_dim)
            c_rep = tf.repeat(c, n_samples, axis=0)   # (n_samples, cond_dim)

            # 3) sample initial x_{t_N} ∼ Normal(0, t_N^2 I)
            t_N = ts[0]
            x = tf.random.normal((n_samples, self.input_dim)) #* t_N

            # 4) DDIM loop
            for n in range(len(ts)-1):
                t_n     = ts[n]     # current time
                t_prev  = ts[n+1]   # next (smaller) time
                t_p = 0.9*ts[n][:, None, None]

                # ← predict clean x₀:
                x0_pred = self.consistency_function(
                    x, c_rep, 
                    tf.fill((n_samples,1), t_n)
                )
                
                s = tf.random.normal((n_samples, self.img_size, self.img_size, 3))
                x = x0_pred + t_p * s

                # ← compute σ_{n-1} and the “α” coefficient a = sqrt((t_prev² − σ²)/t_n²)
                sigma = eta * tf.sqrt(
                    (t_p**2 - t_prev**2) * (t_prev**2) / (t_p**2)
                )
                a = tf.sqrt((t_prev**2 - sigma**2) / (t_p**2))

                # ← the DDIM mean:
                x_mean = a[:, None, None] * x + (1.0 - a)[:, None, None] * x0_pred

                # ← add noise (if η>0)
                if eta > 0:
                    z = tf.random.normal((n_samples, self.img_size, self.img_size, 3))
                    x = x_mean + sigma[:, None, None] * z
                else:
                    x = x_mean

            out.append(x)

        # stack and possibly squeeze
        post = tf.stack(out, axis=0)  # (n_data, n_samples, input_dim)
        if n_data == 1:
            post = tf.squeeze(post, 0)

        return post.numpy() if to_numpy else post

      
    def sample(self, input_dict, n_samples, n_steps=10, to_numpy=True, step_size=1e-3, img_size= 32, **kwargs):
        """Generates random draws from the approximate posterior given a dictionary with conditonal variables
        using the multistep sampling algorithm (Algorithm 1).

        Parameters
        ----------
        input_dict  : dict
            Input dictionary containing the following mandatory keys, if ``DEFAULT_KEYS`` unchanged:
            ``summary_conditions`` : the conditioning variables (including data) that are first passed through a summary network
            ``direct_conditions``  : the conditioning variables that the directly passed to the inference network
        n_samples   : int
            The number of posterior draws (samples) to obtain from the approximate posterior
        n_steps     : int
            The number of sampling steps
        TODO: This does not seem to work in some cases
        to_numpy    : bool, optional, default: True
            Flag indicating whether to return the samples as a ``np.ndarray`` or a ``tf.Tensor``
        **kwargs    : dict, optional, default: {}
            Additional keyword arguments passed to the networks

        Returns
        -------
        post_samples : tf.Tensor or np.ndarray of shape (n_data_sets, n_samples, n_params)
            The sampled parameters from the approximate posterior of each data set
        """

        self.img_size = img_size

        # Compute condition (direct, summary, or both)
        _, conditions = self._compute_summary_condition(
            input_dict.get(defaults.DEFAULT_KEYS["summary_conditions"]),
            input_dict.get(defaults.DEFAULT_KEYS["direct_conditions"]),
            training=False,
            **kwargs,
        )
        n_data_sets, condition_dim = tf.shape(conditions)

        assert condition_dim == self.condition_dim

        post_samples = np.empty(shape=(n_data_sets, n_samples, self.img_size, self.img_size, 3), dtype=np.float32)
        n_data_sets, condition_dim = conditions.shape

        for i in range(n_data_sets):
            c = conditions[i, None]
            c_rep = tf.concat([c] * n_samples, axis=0)
            discretized_time = tf.reverse(discretize_time(self.eps, self.T_max, n_steps), axis=[-1])
            z_init = tf.random.normal((n_samples, self.img_size, self.img_size, 3), stddev=self.T_max)
            T = discretized_time[0] + tf.zeros((n_samples, 1))
            samples = self.consistency_function(z_init, c_rep, T)
            for n in range(1, n_steps):
                z = tf.random.normal((n_samples, self.img_size, self.img_size, 3))
                x_n = samples + tf.math.sqrt(discretized_time[n] ** 2 - self.eps**2) * z
                samples = self.consistency_function(x_n, c_rep, discretized_time[n] + tf.zeros((n_samples, 1)))
            post_samples[i] = samples

        # Remove trailing first dimension in the single data case
        if n_data_sets == 1:
            post_samples = tf.squeeze(post_samples, axis=0)

        # Return numpy version of tensor or tensor itself
        if to_numpy:
            return post_samples.numpy()
        return post_samples

    def _compute_summary_condition(self, summary_conditions, direct_conditions, **kwargs):
        """Determines how to concatenate the provided conditions."""
        
        #s = summary_conditions
        # Compute learnable summaries, if given
        if self.summary_net is not None:
            sum_condition = self.summary_net(summary_conditions, **kwargs)
        else:
            sum_condition = None

        # Concatenate learnable summaries with fixed summaries
        if sum_condition is not None and direct_conditions is not None:
            full_cond = tf.concat([sum_condition, direct_conditions], axis=-1)
        elif sum_condition is not None:
            full_cond = sum_condition
        elif direct_conditions is not None:
            full_cond = direct_conditions
        else:
            raise SummaryStatsError("Could not concatenarte or determine conditioning inputs...")
        return sum_condition, full_cond 

    def _determine_summary_loss(self, loss_fun):
        """Determines which summary loss to use if default `None` argument provided, otherwise return identity."""

        # If callable, return provided loss
        if loss_fun is None or callable(loss_fun):
            return loss_fun

        # If string, check for MMD or mmd
        elif isinstance(loss_fun, str):
            if loss_fun.lower() == "mmd":
                return mmd_summary_space
            else:
                raise NotImplementedError("For now, only 'mmd' is supported as a string argument for summary_loss_fun!")
        # Throw if loss type unexpected
        else:
            raise NotImplementedError(
                "Could not infer summary_loss_fun, argument should be of type (None, callable, or str)!"
            )

    def _determine_loss(self, loss_fun):
        """Determines which summary loss to use if default ``None`` argument provided, otherwise return identity."""

        if loss_fun is None:
            return tf.keras.losses.log_cosh
        return loss_fun

    @classmethod
    def _schedule_discretization(cls, k, K, s0=2.0, s1=100.0):
        """Schedule function for adjusting the discretization level `N` during the course
        of training. Implements the function N(k) from https://arxiv.org/abs/2310.14189,
        Section 3.4.

        Parameters:
        -----------
        k   : int
            Current iteration index.
        K   : int
            Final iteration index (len(dataset) * num_epochs)
        s0  : int, optional, default: 2
            The initial discretization steps
        s1  : int, optional, default: 100
            The final discretization steps
        """
        K_ = tf.floor(K / (tf.math.log(s1 / s0) / tf.math.log(2.0) + 1.0))
        out = tf.minimum(s0 * tf.pow(2.0, tf.floor(k / K_)), s1) + 1.0
        return tf.cast(out, tf.int32)


class DriftNetwork(tf.keras.Model):
    """Implements a learnable velocity field for a neural ODE. Will typically be used
    in conjunction with a ``RectifyingFlow`` instance, as proposed by [1] in the context
    of unconditional image generation.

    [1] Liu, X., Gong, C., & Liu, Q. (2022).
    Flow straight and fast: Learning to generate and transfer data with rectified flow.
    arXiv preprint arXiv:2209.03003.
    """

    def __init__(
        self,
        input_dim,
        cond_dim,
        hidden_dim=512,
        num_hidden=2,
        activation="relu",
        residual_connections=True,
        dropout_rate=0.0,
        kernel_regularization=0.0,
        **kwargs,
    ):
        """Creates a learnable velocity field instance to be used in the context of rectifying
        flows or neural ODEs.

        [1] Liu, X., Gong, C., & Liu, Q. (2022).
        Flow straight and fast: Learning to generate and transfer data with rectified flow.
        arXiv preprint arXiv:2209.03003.

        Parameters
        ----------
        input_dim : int
            The input dimensionality
        cond_dim  : int
            The dimensionality of the condition
        hidden_dim: int, optional, default: 512
            The dimensionality of the hidden layers
        num_hidden: int, optional, default: 2
            The number of hidden layers (minimum 1)
        eps       : float, optional, default: 0.002
            The minimum time
        activation: string, optional, default: 'relu'
            The activation function of the dense layers
        residual_connections: bool, optional, default: True
            Use residual connections in the MLP
        dropout_rate        : float, optional, default: 0.0
            Dropout rate for the hidden layers in the MLP
        kernel_regularization: float, optional, default: 0.0
            L2 regularization factor for the kernel weights
        """

        super().__init__(**kwargs)

        # set for compatibility with RectifiedDistribution
        self.latent_dim = input_dim
        self.net = ConfigurableMLP(
            input_dim=input_dim,
            condition_dim=cond_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            activation=activation,
            residual_connections=residual_connections,
            dropout_rate=dropout_rate,
            kernel_regularization=kernel_regularization,
        )
        self.net.build(input_shape=())

    def call(self, target_vars, latent_vars, time, condition, **kwargs):
        """Performs a linear interpolation between target and latent variables
        over time (i.e., a single ODE step during training).

        Parameters
        ----------
        target_vars : tf.Tensor of shape (batch_size, ..., num_targets)
            The variables of interest (e.g., parameters) over which we perform inference.
        latent_vars : tf.Tensor of shape (batch_size, ..., num_targets)
            The sampled random variates from the base distribution.
        time        : tf.Tensor of shape (batch_size, ..., 1)
            A vector of time indices in (0, 1)
        condition   : tf.Tensor of shape (batch_size, ..., condition_dim)
            The optional conditioning variables (e.g., as returned by a summary network)
        **kwargs    : dict, optional, default: {}
            Optional keyword arguments passed to the ``tf.keras.Model`` call() method
        """

        diff = target_vars - latent_vars
        wdiff = time * target_vars + (1 - time) * latent_vars
        drift = self.drift(wdiff, time, condition, **kwargs)
        return diff, drift

    def drift(self, target_t, time, condition, **kwargs):
        """Returns the drift at target_t time given optional condition(s).

        Parameters
        ----------
        target_t    : tf.Tensor of shape (batch_size, ..., num_targets)
            The variables of interest (e.g., parameters) over which we perform inference.
        time        : tf.Tensor of shape (batch_size, ..., 1)
            A vector of time indices in (0, 1)
        condition   : tf.Tensor of shape (batch_size, ..., condition_dim)
            The optional conditioning variables (e.g., as returned by a summary network)
        **kwargs    : dict, optional, default: {}
            Optional keyword arguments passed to the drift network.
        """

        if condition is not None:
            inp = [target_t, condition, time]
        else:
            inp = [target_t, time]
        return self.net(inp, **kwargs)
