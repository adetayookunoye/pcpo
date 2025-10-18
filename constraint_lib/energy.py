
import jax.numpy as jnp

def kinetic_energy(u, v):
    return jnp.mean(0.5 * (u**2 + v**2), axis=(-2, -1))

def energy_error(u_pred, v_pred, u_ref, v_ref):
    ke_p = kinetic_energy(u_pred, v_pred)
    ke_r = kinetic_energy(u_ref, v_ref)
    # smooth relative squared error for stable gradients
    rel = (ke_p - ke_r) / (ke_r + 1e-8)
    return jnp.mean(rel**2)
