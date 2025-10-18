
import jax
import jax.numpy as jnp
import numpy as np

from constraint_lib.divergence_free import psi_to_uv

def smooth_noise(key, shape, sigma=3.0):
    # FFT-based smoothing of white noise to get coherent streamfunction
    noise = jax.random.normal(key, shape)
    kx = jnp.fft.fftfreq(shape[-1])[:, None]
    ky = jnp.fft.fftfreq(shape[-2])[None, :]
    k2 = kx**2 + ky**2
    filt = jnp.exp(-0.5 * k2 * sigma**2)
    spec = jnp.fft.fft2(noise)
    sm = jnp.fft.ifft2(spec * filt).real
    return sm

def generate_batch(key, batch_size=8, H=64, W=64, steps=5, return_seq=False):
    # Generate initial psi, derive velocity, and create trivial time rollout by slight diffusion decay.
    keys = jax.random.split(key, batch_size)
    psi0 = jnp.stack([smooth_noise(k, (H,W)) for k in keys], axis=0)  # (B,H,W)
    uv0 = psi_to_uv(psi0)  # (B,2,H,W)
    # Rollout: exponential decay to emulate dissipative behavior
    seq = []
    for t in range(steps):
        decay = jnp.exp(-0.1 * t)
        uv_t = uv0 * decay
        seq.append(jnp.moveaxis(uv_t, 1, -1))  # (B,H,W,2)
    x = seq[0]
    y = seq[-1]
    if return_seq:
        seq_arr = jnp.stack(seq, axis=1)  # (B, steps, H, W, 2)
        return x, y, seq_arr
    return x, y
