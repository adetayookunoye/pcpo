
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from models.fno import FNO2d
from constraint_lib.divergence_free import psi_to_uv

class Encoder(eqx.Module):
    w1: jnp.ndarray
    w2_mu: jnp.ndarray
    w2_logvar: jnp.ndarray

    def __init__(self, in_ch, latent_dim, key):
        k1,k2,k3 = jax.random.split(key,3)
        self.w1 = jax.random.normal(k1,(in_ch,128))*0.02
        self.w2_mu = jax.random.normal(k2,(128,latent_dim))*0.02
        self.w2_logvar = jax.random.normal(k3,(128,latent_dim))*0.02

    def __call__(self, x):
        # x: (B,H,W,C)
        x = jnp.mean(x, axis=(1,2))  # global avg
        h = jax.nn.gelu(jnp.dot(x, self.w1))
        mu = jnp.dot(h, self.w2_mu)
        logvar = jnp.dot(h, self.w2_logvar)
        return mu, logvar

class CVAEFNO(eqx.Module):
    enc: Encoder
    dec: FNO2d
    beta: float = 1.0
    dx: float = 1.0
    dy: float = 1.0

    def __init__(self, in_ch=2, latent_dim=16, width=32, modes=12, depth=4, beta=1.0, key=jax.random.PRNGKey(0)):
        k1,k2 = jax.random.split(key,2)
        self.enc = Encoder(in_ch, latent_dim, k1)
        # Decoder takes (velocity + latent) as channels
        self.dec = FNO2d(in_ch=in_ch+latent_dim, out_ch=1, width=width, modes=modes, depth=depth, key=k2)
        self.beta = beta

    def reparam(self, mu, logvar, key):
        eps = jax.random.normal(key, mu.shape)
        std = jnp.exp(0.5 * logvar)
        return mu + eps * std

    def __call__(self, x, key):
        mu, logvar = self.enc(x)
        z = self.reparam(mu, logvar, key)
        # Broadcast z over spatial dims
        B,H,W,C = x.shape
        z_img = jnp.tile(z[:,None,None,:], (1,H,W,1))
        xz = jnp.concatenate([x, z_img], axis=-1)
        psi = self.dec(xz)[...,0]
        uv = psi_to_uv(psi, dx=self.dx, dy=self.dy)  # (B,2,H,W)
        uv = jnp.moveaxis(uv, 1, -1)  # (B,H,W,2)
        return uv, mu, logvar
