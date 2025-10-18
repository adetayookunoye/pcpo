
import jax
import jax.numpy as jnp
import equinox as eqx
from models.fno import FNO2d
from constraint_lib.divergence_free import psi_to_uv

class DivFreeFNO(eqx.Module):
    fno: FNO2d
    dx: float = 1.0
    dy: float = 1.0

    def __init__(self, width=32, modes=12, depth=4, key=jax.random.PRNGKey(0)):
        self.fno = FNO2d(in_ch=2, out_ch=1, width=width, modes=modes, depth=depth, key=key)

    def __call__(self, x):
        # x: (B,H,W,2) velocity input at t0
        psi = self.fno(x)  # (B,H,W,1)
        psi = psi[..., 0]
        uv = psi_to_uv(psi, dx=self.dx, dy=self.dy)  # (B,2,H,W)
        # Return (B,H,W,2)
        return jnp.moveaxis(uv, 1, -1)
