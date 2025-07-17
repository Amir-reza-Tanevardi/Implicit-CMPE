import argparse
import datetime
import os
import importlib.util
import pickle
import sys

import numpy as np
import tensorflow as tf
from bayesflow.experimental.rectifiers import RectifiedDistribution
from bayesflow.trainers import Trainer
from scipy.ndimage import gaussian_filter
from skimage.util import random_noise
from tensorflow import keras
from tensorflow.keras import layers

sys.path.append("../../")
from amortizers import ConsistencyAmortizer

# --- Data Preprocessing ---

def load_imagenet(img_size, split):
    """Loads ImageNet using TensorFlow Datasets."""
    import tensorflow_datasets as tfds
    def _preprocess(image, label):
        image = tf.image.resize(image, [img_size, img_size])
        image = tf.cast(image, tf.float32) / 255.0  # [0,1]
        image = image * 2.0 - 1.0  # [-1,1]
        return image, image

    ds = tfds.load('imagenette', split=split, as_supervised=True, data_dir = "/work/pi_aghasemi_umass_edu/afzali_umass/W2S/.cache")
    ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    #ds = ds.shuffle(1024).prefetch(tf.data.AUTOTUNE)
    return ds

# --- Masking and Corruption ---

def inpainting_mask(image, mask_size=32):
    # image: numpy array HxWx3 in [-1,1]
    H, W, C = image.shape
    top = np.random.randint(0, H - mask_size)
    left = np.random.randint(0, W - mask_size)
    masked = image.copy()
    masked[top:top + mask_size, left:left + mask_size, :] = 0.0
    return masked


def grayscale_camera_rgb(theta, noise='poisson', psf_width=2.5, noise_scale=1.0, noise_gain=0.5):
    # Apply noise+blur channel-wise
    noisy = noise_gain * random_noise(noise_scale * theta, mode=noise)
    blurred = np.stack([gaussian_filter(noisy[..., c], sigma=psf_width) for c in range(3)], axis=-1)
    return blurred

# --- Configurators ---

def configurator_blurred(f):
    """
    Parameters:
    f['prior_draws']: numpy array (B, H, W, C)
    f['sim_data']: numpy array (B, H, W, C)
    Returns:
     - parameters: flattened prior_draws (B, H*W*C)
     - summary_conditions: blurred sim_data (B, H, W, C)
    """
    B, H, W, C = f['prior_draws'].shape
    # Flatten parameters (already normalized)
    p = f['prior_draws'].reshape(B, -1).astype(np.float32)
    # Create blurred summary conditions
    blurred = np.stack([grayscale_camera_rgb(f['sim_data'][b]) for b in range(B)], axis=0).astype(np.float32)
    return {'parameters': p, 'summary_conditions': blurred}


def configurator_masked(f):
    """
    Applies an inpainting mask to sim_data instead of blur.
    Returns masked images as summary_conditions.
    """
    B, H, W, C = f['prior_draws'].shape
    p = f['prior_draws'].reshape(B, -1).astype(np.float32)
    masked = np.stack([inpainting_mask(f['sim_data'][b]) for b in range(B)], axis=0).astype(np.float32)
    return {'parameters': p, 'summary_conditions': masked}

# --- Network Blocks ---

def kernel_init(scale):
    scale = max(scale, 1e-10)
    return keras.initializers.VarianceScaling(scale, mode='fan_avg', distribution='uniform')

class AttentionBlock(layers.Layer):
    def __init__(self, units, groups=8, **kwargs):
        super().__init__(**kwargs)
        self.norm = layers.GroupNormalization(groups)
        self.query = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.key   = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.value = layers.Dense(units, kernel_initializer=kernel_init(1.0))
        self.proj  = layers.Dense(units, kernel_initializer=kernel_init(0.0))

    def call(self, x):
        b, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x_norm = self.norm(x)

        q = self.query(x_norm)  # (b, h, w, c)
        k = self.key(x_norm)
        v = self.value(x_norm)

        # Flatten spatial dimensions
        q_flat = tf.reshape(q, [b, h * w, c])  # (b, N, c)
        k_flat = tf.reshape(k, [b, h * w, c])
        v_flat = tf.reshape(v, [b, h * w, c])

        # Compute scaled dot-product attention
        scale = tf.cast(c, tf.float32) ** -0.5
        attn_weights = tf.matmul(q_flat, k_flat, transpose_b=True) * scale  # (b, N, N)
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        # Attention output
        attn_out = tf.matmul(attn_weights, v_flat)  # (b, N, c)
        attn_out = tf.reshape(attn_out, [b, h, w, c])
        out = self.proj(attn_out)
        return x + out


class TimeEmbedding(layers.Layer):
    def __init__(self, dim, tmax, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim; self.tmax = tmax
        half = dim // 2
        emb = tf.range(half, dtype=tf.float32)
        self.inv_freq = tf.exp(-tf.math.log(10000.0) * emb / (half - 1))
    def call(self, t):
        t = tf.cast(t, tf.float32) * (1000.0 / self.tmax)
        sin = tf.sin(tf.expand_dims(t, -1) * self.inv_freq)
        cos = tf.cos(tf.expand_dims(t, -1) * self.inv_freq)
        return tf.concat([sin, cos], -1)

# ResidualBlock, DownSample, UpSample, TimeMLP implementations omitted for brevity

def ResidualBlock(width, groups=8, activation_fn=keras.activations.swish):
    def apply(inputs):
        x, t = inputs
        input_width = x.shape[3]

        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1, kernel_initializer=kernel_init(1.0))(x)

        temb = activation_fn(t)
        temb = layers.Dense(width, kernel_initializer=kernel_init(1.0))(temb)[:, None, None, :]

        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0))(x)

        x = layers.Add()([x, temb])
        x = layers.GroupNormalization(groups=groups)(x)
        x = activation_fn(x)

        x = layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(0.0))(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownSample(width):
    def apply(x):
        x = layers.Conv2D(
            width,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer=kernel_init(1.0),
        )(x)
        return x

    return apply


def UpSample(width, interpolation="nearest"):
    def apply(x):
        x = layers.UpSampling2D(size=2, interpolation=interpolation)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", kernel_initializer=kernel_init(1.0))(x)
        return x

    return apply


def TimeMLP(units, activation_fn=keras.activations.swish):
    def apply(inputs):
        temb = layers.Dense(units, activation=activation_fn, kernel_initializer=kernel_init(1.0))(inputs)
        temb = layers.Dense(units, kernel_initializer=kernel_init(1.0))(temb)
        return temb

    return apply


# --- U-Net Model ---

def build_unet_model(img_size, channels, widths, has_attention, tmax,
                     cond_dim=128, num_res_blocks=2, norm_groups=8,
                     first_channels=64):
    image_input = layers.Input(shape=(img_size*img_size*channels,), name='image_input')
    x = layers.Reshape((img_size, img_size, channels))(image_input)
    time_input = layers.Input(shape=(), name='time_input', dtype=tf.float32)
    cond_input = layers.Input(shape=(cond_dim,), name='condition_input')

    # Initial conv
    x0 = layers.Conv2D(first_channels, 3, padding='same', kernel_initializer=kernel_init(1.0))(x)

    # Time embedding
    temb = TimeEmbedding(first_channels*4, tmax)(time_input)
    temb = layers.Dense(first_channels*4, activation='swish')(temb)
    temb = layers.Dense(first_channels*4)(temb)
    temb = layers.Concatenate()([temb, cond_input])

    # Down path
    skips = [x0]
    x = x0
    for i, w in enumerate(widths):
        for _ in range(num_res_blocks):
            x = ResidualBlock(w, norm_groups)([x, temb])
            if has_attention[i]: x = AttentionBlock(w, norm_groups)(x)
            skips.append(x)
        if i < len(widths)-1:
            x = DownSample(width=w)(x)
            skips.append(x)

    # Middle
    x = ResidualBlock(widths[-1], norm_groups)([x, temb])
    x = AttentionBlock(widths[-1], norm_groups)(x)
    x = ResidualBlock(widths[-1], norm_groups)([x, temb])

    # Up path
    for i in reversed(range(len(widths))):
        for _ in range(num_res_blocks+1):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(widths[i], norm_groups)([x, temb])
            if has_attention[i]: x = AttentionBlock(widths[i], norm_groups)(x)
        if i>0:
            x = UpSample(widths[i])(x)

    # Output conv
    x = layers.GroupNormalization(norm_groups)(x)
    x = layers.Activation('swish')(x)
    x = layers.Conv2D(channels, 3, padding='same', kernel_initializer=kernel_init(0.0))(x)
    out = x

    model = keras.Model([image_input, cond_input, time_input], out, name='imagenet_unet')
    model.latent_dim = img_size*img_size*channels
    model.input_dim = img_size*img_size*channels
    model.condition_dim = cond_dim
    return model

# --- Trainer Setup ---

def build_trainer(args, forward_train=None):
    img_size = args.img_size
    channels = 3
    cond_dim = 128

    # Build summary network
    summary_net = keras.Sequential([
        layers.Conv2D(64,3,activation='relu',padding='same', input_shape=(img_size,img_size,channels)),
        layers.GroupNormalization(args.norm_groups),
        layers.Conv2D(64,3,activation='relu',padding='same'),
        layers.GroupNormalization(args.norm_groups),
        layers.Conv2D(128,3,activation='relu',padding='same'),
        layers.GroupNormalization(args.norm_groups),
        layers.Conv2D(128,3,activation='relu',padding='same'),
        layers.GroupNormalization(args.norm_groups),
        layers.GlobalAveragePooling2D()
    ])

    # U-Net
    widths = [64, 128, 256]
    has_attention = [False, True, True]
    unet = build_unet_model(img_size, channels, widths, has_attention,
                            tmax=args.tmax, cond_dim=cond_dim,
                            num_res_blocks=args.res_blocks,
                            norm_groups=args.norm_groups,
                            first_channels=args.base_channels)

    
    batch_size = args.batch_size
    num_steps = args.num_steps
    initial_learning_rate = args.initial_learning_rate
    if forward_train is not None:
        num_batches = np.ceil(forward_train["prior_draws"].shape[0] / batch_size)
        num_epochs = int(np.ceil(num_steps / num_batches))
        num_steps = num_epochs * num_batches
    else:
        num_epochs = 0
        num_steps = 0
    
    
    if args.fine_tune_summary:
        summary_net.trainable = False



    amortizer = ConsistencyAmortizer(
        consistency_net=unet,
        summary_net=summary_net,
        num_steps=args.num_steps,
        sigma2=args.sigma2,
        eps=args.epsilon,
        T_max=args.tmax,
        s0=args.s0,
        s1=args.s1
    )

    trainer = Trainer(amortizer, configurator=configurator_blurred,
                      checkpoint_path=args.checkpoint_path)

    if forward_train is not None:
        # Optimizer
        if args.lr_adapt == "cosine":
            lr = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate, num_steps)
        elif args.lr_adapt == "none":
            lr = args.initial_learning_rate
        else:
            raise ValueError(f"Invalid value for learning rate adaptation: '{args.lr_adapt}'")

        if args.optimizer.lower() == "adamw":
            optimizer = tf.keras.optimizers.AdamW(lr)
        else:
            optimizer = type(tf.keras.optimizers.get(args.optimizer))(lr)

        return trainer, optimizer, num_epochs, batch_size

    return trainer

# --- Main Script ---

if __name__=='__main__':
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except (ValueError, RuntimeError):
            # Invalid device or cannot modify virtual devices once initialized.
            pass
    
    ##### Fix to bayesflow helper_functions.py file #####
    spec = importlib.util.find_spec("bayesflow")
    if spec and spec.origin:
        bayesflow_root = os.path.dirname(spec.origin)  # path to bayesflow package
        target_file = os.path.join(bayesflow_root, "helper_functions.py")  # adjust as needed

        # Example: patch the file
        with open(target_file, "r") as f:
            lines = f.readlines()

        # Replace deprecated code or make other fixes
        lines = [line.replace("optimizer.lr", "optimizer.learning_rate") for line in lines]  # example fix

        with open(target_file, "w") as f:
            f.writelines(lines)

        print(f"Patched {target_file}")
    else:
        print("Could not locate bayesflow.")
    #####################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--initial-learning-rate', type=float, default=5e-4)
    parser.add_argument('--num-steps', type=int, default=100000)
    parser.add_argument("--num-training", type=int, default=12000)
    parser.add_argument("--lr-adapt", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument('--tmax', type=float, default=1000.0)
    parser.add_argument('--epsilon', type=float, default=1e-3)
    parser.add_argument('--s0', type=int, default=10)
    parser.add_argument('--s1', type=int, default=50)
    parser.add_argument('--sigma2', type=float, default=0.25)
    parser.add_argument('--res-blocks', type=int, default=2)
    parser.add_argument('--norm-groups', type=int, default=8)
    parser.add_argument('--base-channels', type=int, default=64)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument('--fine-tune-summary', action='store_true')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoints/imagenet-unet-deblurring')
    parser.add_argument("--method", type=str, default="cmpe", choices=["cmpe", "fmpe"])
    parser.add_argument("--architecture", type=str, default="unet", choices=["unet", "naive"])
    args = parser.parse_args()

    os.makedirs(args.checkpoint_path, exist_ok=True)
    with open(os.path.join(args.checkpoint_path,'args.pkl'),'wb') as f:
        pickle.dump(vars(args), f)
    
    import os, psutil
    process = psutil.Process(os.getpid())
    print("CPU RAM usage (GB):", process.memory_info().rss / 1e9)
    
    train_ds = load_imagenet(args.img_size, 'train')
    val_ds   = load_imagenet(args.img_size, 'validation')
    
    train_ds_unbatched = train_ds.take(args.num_training)
    val_ds_unbatched = val_ds.take(10)
    
    train_imgs = []
    train_lbls = []
    for img, lbl in train_ds_unbatched:
        train_imgs.append(img.numpy())
        train_lbls.append(lbl.numpy())
    train_imgs = np.stack(train_imgs, axis=0)
    train_lbls = np.stack(train_lbls, axis=0)

    val_imgs = []
    val_lbls = []
    for img, lbl in val_ds_unbatched:
        val_imgs.append(img.numpy())
        val_lbls.append(lbl.numpy())
    val_imgs = np.stack(val_imgs, axis=0)
    val_lbls = np.stack(val_lbls, axis=0)

    forward_train = {'prior_draws': train_imgs,
         'sim_data': train_imgs}
    forward_val = {'prior_draws': val_imgs,
                         'sim_data': val_imgs}

    trainer, optimizer, num_epochs, batch_size = build_trainer(args, forward_train=forward_train)

    print(f"Training for {num_epochs} epochs...")

    process = psutil.Process(os.getpid())
    print("CPU RAM usage (GB):", process.memory_info().rss / 1e9)
    
    trainer.train_offline(
        forward_train,
        optimizer=optimizer,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_sims=forward_val
    )
