
import jax.numpy as jnp
from jax import grad

def psi_to_uv(psi, dx=1.0, dy=1.0):
    """Compute velocities from stream function psi (B,H,W -> B,2,H,W)."""
    # Finite-difference grads (central)
    def d_dx(f):
        return (jnp.roll(f, -1, axis=-1) - jnp.roll(f, 1, axis=-1)) / (2.0 * dx)
    def d_dy(f):
        return (jnp.roll(f, -1, axis=-2) - jnp.roll(f, 1, axis=-2)) / (2.0 * dy)
    u = d_dy(psi)  # ∂ψ/∂y
    v = -d_dx(psi) # -∂ψ/∂x
    return jnp.stack([u, v], axis=1)

def divergence(u, v, dx=1.0, dy=1.0):
    def d_dx(f):
        return (jnp.roll(f, -1, axis=-1) - jnp.roll(f, 1, axis=-1)) / (2.0 * dx)
    def d_dy(f):
        return (jnp.roll(f, -1, axis=-2) - jnp.roll(f, 1, axis=-2)) / (2.0 * dy)
    return d_dx(u) + d_dy(v)

def vorticity(u, v, dx=1.0, dy=1.0):
    du_dy = (jnp.roll(u, -1, axis=-2) - jnp.roll(u, 1, axis=-2)) / (2.0 * dy)
    dv_dx = (jnp.roll(v, -1, axis=-1) - jnp.roll(v, 1, axis=-1)) / (2.0 * dx)
    return dv_dx - du_dy
