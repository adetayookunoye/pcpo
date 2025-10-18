
import jax, jax.numpy as jnp
from models.fno import FNO2d

def test_fno_forward_shape():
    key = jax.random.PRNGKey(0)
    model = FNO2d(in_ch=2, out_ch=1, width=8, modes=4, depth=2, key=key)
    x = jax.random.normal(key, (2,32,32,2))
    y = model(x)
    assert y.shape == (2,32,32,1)
