import argparse
import datetime
import os
import importlib.util
import pickle
import sys
import math

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
    ds = ds.shuffle(1024).prefetch(tf.data.AUTOTUNE)
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
    p = f['prior_draws'].astype(np.float32)
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

# -----------------------------------
# Helper layers & functions
# -----------------------------------
def default_init(scale):
    return keras.initializers.VarianceScaling(scale, mode='fan_avg', distribution='uniform')

def normalization(channels):
    # equivalent to PyTorch nn.GroupNorm(32, channels)
    return layers.LayerNormalization(axis=-1)

def linear(in_dim, out_dim):
    return layers.Dense(out_dim, kernel_initializer=default_init(1.0))

def zero_conv(in_ch, out_ch, kernel_size=3):
    return layers.Conv2D(out_ch, kernel_size, padding='same',
                         kernel_initializer=keras.initializers.Zeros())

def timestep_embedding(timesteps, dim):
    half = dim // 2
    emb = tf.cast(timesteps, tf.float32)[..., None] * tf.exp(
        -math.log(10000) * tf.range(half, dtype=tf.float32) / half
    )
    return tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)

# -----------------------------------
# QKV attention (fallback)
# -----------------------------------
class QKVAttention(layers.Layer):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def call(self, qkv, encoder_kv=None):
        # qkv: [B, 3*C, N], where C = head_dim * n_heads
        B, WC, N = tf.shape(qkv)[0], tf.shape(qkv)[1], tf.shape(qkv)[2]
        C = WC // 3
        head_dim = C // self.n_heads
        q, k, v = tf.split(qkv, 3, axis=1)
        if encoder_kv is not None:
            ek, ev = tf.split(encoder_kv, 2, axis=1)
            k = tf.concat([ek, k], axis=2)
            v = tf.concat([ev, v], axis=2)
        scale = 1.0 / tf.sqrt(tf.sqrt(tf.cast(head_dim, tf.float32)))
        q = tf.reshape(q * scale, [B*self.n_heads, head_dim, N])
        k = tf.reshape(k * scale, [B*self.n_heads, head_dim, -1])
        v = tf.reshape(v, [B * self.n_heads, -1, head_dim])
        weight = tf.nn.softmax(tf.matmul(tf.transpose(q, [0,2,1]), k), axis=-1)  # [B*h, N, M]
        a = tf.matmul(weight, v)  # [B*h, N, head_dim]
        a = tf.reshape(a, [B, C, N])
        return a

# -----------------------------------
# Core Blocks
# -----------------------------------
class ResBlock(layers.Layer):
    def __init__(self, channels, emb_channels, dropout, out_channels=None,
                 use_conv=False, up=False, down=False):
        super().__init__()
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.up, self.down = up, down

        # in_layers
        self.norm1 = normalization(channels)
        self.act1  = layers.Activation('swish')
        self.conv1 = layers.Conv2D(self.out_channels, 3, padding='same',
                                   kernel_initializer=default_init(1.0))

        # up/down
        if up:
            self.h_up = layers.UpSampling2D(interpolation='nearest')
            self.x_up = layers.UpSampling2D(interpolation='nearest')
        elif down:
            self.h_up = layers.AveragePooling2D()
            self.x_up = layers.AveragePooling2D()
        else:
            self.h_up = self.x_up = lambda x: x

        # emb layers
        mlp_units = 2*self.out_channels if False else self.out_channels
        self.emb_layers = keras.Sequential([
            layers.Activation('swish'),
            linear(emb_channels, mlp_units),
        ])

        # out_layers
        self.norm2 = normalization(self.out_channels)
        self.act2  = layers.Activation('swish')
        self.dropout = layers.Dropout(dropout)
        self.conv2 = zero_conv(self.out_channels, self.out_channels)

        # skip
        if channels != self.out_channels:
            if use_conv:
                self.skip = layers.Conv2D(self.out_channels, 3, padding='same',
                                          kernel_initializer=default_init(1.0))
            else:
                self.skip = layers.Conv2D(self.out_channels, 1, padding='same',
                                          kernel_initializer=default_init(1.0))
        else:
            self.skip = lambda x: x

    def call(self, x, emb):
        # x: [B,H,W,C], emb: [B, emb_channels]
        h = self.norm1(x)
        h = self.act1(h)
        if self.up or self.down:
            h = self.h_up(h)
            x = self.x_up(x)
        h = self.conv1(h)

        emb_out = self.emb_layers(emb)  # [B, out_ch] or [B, 2*out_ch]
        emb_out = emb_out[:, None, None, :]
        h = h + emb_out

        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return self.skip(x) + h

class AttentionBlock(layers.Layer):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.norm = normalization(channels)
        self.qkv  = layers.Conv2D(channels*3, 1, kernel_initializer=default_init(1.0))
        self.proj = zero_conv(channels, channels)

        self.attn = QKVAttention(num_heads)

    def call(self, x, encoder_out=None):
        # x: [B,H,W,C]
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], x.shape[3]
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)                          # [B,H,W,3C]
        qkv = tf.reshape(qkv, [B, -1, 3*C])             # [B,N,3C]
        qkv = tf.transpose(qkv, [0,2,1])                # [B,3C,N]
        if encoder_out is None:
            h = self.attn(qkv)
        else:
            # you can pass encoder_kv similarly if needed
            h = self.attn(qkv, encoder_out)
        h = tf.reshape(tf.transpose(h, [0,2,1]), [B,H,W,C])
        return x + self.proj(h)

class Downsample(layers.Layer):
    def __init__(self, channels, use_conv=True):
        super().__init__()
        if use_conv:
            self.op = layers.Conv2D(channels, 3, strides=2, padding='same',
                                    kernel_initializer=default_init(1.0))
        else:
            self.op = layers.AveragePooling2D()

    def call(self, x):
        return self.op(x)

class Upsample(layers.Layer):
    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.ups = layers.UpSampling2D(interpolation='nearest')
        if use_conv:
            self.conv = layers.Conv2D(channels, 3, padding='same',
                                      kernel_initializer=default_init(1.0))
        else:
            self.conv = lambda x: x

    def call(self, x):
        return self.conv(self.ups(x))

# -----------------------------------
# The UNetModel
# -----------------------------------
class UNetModel(keras.Model):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0.0,
        channel_mult=(1,2,4,8),
        use_conv_resample=True,
        cond_dim=None,           # like num_classes
        **kwargs
    ):
        super().__init__(**kwargs)
        # timestep + cond embeddings
        time_embed_dim = model_channels * 4
        self.time_embed = keras.Sequential([
            linear(model_channels, time_embed_dim),
            layers.Activation('swish'),
            linear(time_embed_dim, time_embed_dim),
        ])
        if cond_dim is not None:
            self.cond_proj = linear(cond_dim, time_embed_dim)
        else:
            self.cond_proj = None

        self.input_dim = image_size*image_size*in_channels
        self.latent_dim = image_size*image_size*in_channels
        self.condition_dim = cond_dim
        
        # input conv
        self.init_conv = layers.Conv2D(model_channels, 3, padding='same',
                                       kernel_initializer=default_init(1.0))

        # build encoder blocks
        self.input_blocks = []
        ch = model_channels * channel_mult[0]
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                self.input_blocks.append(
                    ResBlock(ch, time_embed_dim, dropout,
                             out_channels=mult*model_channels,
                             use_conv=use_conv_resample)
                )
                ch = mult * model_channels
                if ds in attention_resolutions:
                    self.input_blocks.append(AttentionBlock(ch, num_heads=1))
                input_block_chans.append(ch)        
            if level != len(channel_mult)-1:
                self.input_blocks.append(Downsample(ch, use_conv_resample))
                input_block_chans.append(ch)
                ds *= 2

        # middle
        self.middle = [
            ResBlock(ch, time_embed_dim, dropout),
            AttentionBlock(ch, num_heads=1),
            ResBlock(ch, time_embed_dim, dropout),
        ]

        # decoder
        self.output_blocks = []
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks+1):
                ich = input_block_chans.pop()
                print(f"mult*model_channels in decoder: {mult*model_channels}")
                print(f"ch + ich: {ch + ich}")
                self.output_blocks.append(
                    ResBlock(ch + ich, time_embed_dim, dropout,
                             out_channels=mult*model_channels,
                             use_conv=use_conv_resample)
                )
                ch = mult * model_channels
                if ds in attention_resolutions:
                    self.output_blocks.append(AttentionBlock(ch, num_heads=1))
            if level != 0:
                self.output_blocks.append(Upsample(ch, use_conv_resample))
                ds //= 2

        # final
        self.out_norm = normalization(ch)
        self.out_act  = layers.Activation('swish')
        self.out = zero_conv(ch, out_channels)

    def call(self, inputs):
        x, cond_input, timesteps = inputs
        # x: [B,H,W,C], timesteps: [B], cond_input: [B,cond_dim]
        # embed time + cond
        emb = self.time_embed(timesteps)                 # [B, Tdim]
        if self.cond_proj is not None:
            emb = emb + self.cond_proj(cond_input)

        h = self.init_conv(x)
        hs = [h]

        # encode
        for layer in self.input_blocks:
            if isinstance(layer, ResBlock):
                h = layer(h, emb)
                hs.append(h)
                print(f"h in input: {h.shape}")
                print(f"hs in input: {len(hs)}")
                print("")
            else:
                h = layer(h)
            
        # middle
        for layer in self.middle:
            if isinstance(layer, ResBlock):
                h = layer(h, emb)
            else:
                h = layer(h)

        # decode
        for layer in self.output_blocks:
            if isinstance(layer, ResBlock):
                skip = hs.pop()
                if h.shape[1] != skip.shape[1] or h.shape[2] != skip.shape[2]:
                    raise ValueError(f"Shape mismatch in skip connection: {h.shape} vs {skip.shape}")
                print(f"h in output: {h.shape}") 
                print(f"skip in output: {skip.shape}")
                print("")
                h = tf.concat([h, skip], axis=-1)
                h = layer(h, emb)
            else:
                h = layer(h)

        # final
        h = self.out_norm(h)
        h = self.out_act(h)
        return self.out(h)

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
    # unet = build_unet_model(img_size, channels, widths, has_attention,
    #                         tmax=args.tmax, cond_dim=cond_dim,
    #                         num_res_blocks=args.res_blocks,
    #                         norm_groups=args.norm_groups,
    #                         first_channels=args.base_channels)

    unet = UNetModel(image_size = img_size,
        in_channels = channels,
        model_channels = args.base_channels,
        out_channels = 3,
        num_res_blocks = args.res_blocks,
        attention_resolutions = [32,16,8],
        dropout=0.0,
        channel_mult=(1, 1, 2, 2, 4, 4),
        use_conv_resample=True,
        cond_dim=cond_dim)   

    
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
    parser.add_argument('--img-size', type=int, default=256)
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
    parser.add_argument('--base-channels', type=int, default=128)
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
