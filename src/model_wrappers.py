import equinox as eqx
import jax
import jax.numpy as jnp


def _softplus_inverse(x: float) -> float:
    """Numerically stable inverse softplus for scalar constants."""
    x = float(x)
    return jnp.log(jnp.expm1(x) + 1e-8)


class AmplitudeWrapper(eqx.Module):
    base: eqx.Module
    log_gain: jnp.ndarray
    reg_weight: float
    enabled: bool

    def __init__(
        self,
        base: eqx.Module,
        init_gain: float = 1.0,
        reg_weight: float = 1e-6,
        enabled: bool = True,
    ):
        object.__setattr__(self, "base", base)
        init_value = jnp.asarray(_softplus_inverse(init_gain), dtype=jnp.float32)
        object.__setattr__(self, "log_gain", init_value)
        object.__setattr__(self, "reg_weight", float(reg_weight))
        object.__setattr__(self, "enabled", bool(enabled))

    def gain(self) -> jnp.ndarray:
        if not self.enabled:
            return jnp.array(1.0, dtype=jnp.float32)
        return jax.nn.softplus(self.log_gain)

    def amplitude_regularizer(self) -> jnp.ndarray:
        if not self.enabled or self.reg_weight <= 0.0:
            return jnp.array(0.0, dtype=jnp.float32)
        g = self.gain()
        return self.reg_weight * (g - 1.0) ** 2

    def _apply_gain(self, outputs, gain):
        if isinstance(outputs, tuple):
            if not outputs:
                return outputs
            scaled_first = gain * outputs[0]
            return (scaled_first, *outputs[1:])
        return gain * outputs

    def __call__(self, *args, **kwargs):
        return self._apply_gain(self.base(*args, **kwargs), self.gain())

    def physics_loss(self, *args, **kwargs):
        return self.base.physics_loss(*args, **kwargs)

    def __getattr__(self, name):
        if name in {"base", "log_gain"}:
            return object.__getattribute__(self, name)
        return getattr(self.base, name)
