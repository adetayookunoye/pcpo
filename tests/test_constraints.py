
import jax, jax.numpy as jnp
from constraint_lib.divergence_free import psi_to_uv, divergence

def test_divfree_from_psi():
    key = jax.random.PRNGKey(0)
    psi = jax.random.normal(key, (2,16,16))
    uv = psi_to_uv(psi)
    div = divergence(uv[:,0], uv[:,1])
    assert jnp.abs(div).mean() < 1e-1  # coarse grid tolerance
