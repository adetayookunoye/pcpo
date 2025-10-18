# Bayesian DeepONet implementation
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, Optional
import math

class BranchNet(eqx.Module):
    """Branch network for DeepONet - processes input functions"""
    layers: list
    
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, key: jax.random.PRNGKey):
        keys = jax.random.split(key, len(hidden_dims) + 1)
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = []
        
        for i in range(len(dims) - 1):
            linear = eqx.nn.Linear(dims[i], dims[i+1], key=keys[i])
            self.layers.append(linear)
            if i < len(dims) - 2:  # No activation on output
                self.layers.append(jax.nn.swish)
    
    def _forward(self, x: jnp.ndarray) -> jnp.ndarray:
        y = x
        for layer in self.layers:
            y = layer(y)
        return y

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim > 1:
            return jax.vmap(self._forward)(x)
        return self._forward(x)

class TrunkNet(eqx.Module):
    """Trunk network for DeepONet - processes spatial coordinates"""
    layers: list
    
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, key: jax.random.PRNGKey):
        keys = jax.random.split(key, len(hidden_dims) + 1)
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = []
        
        for i in range(len(dims) - 1):
            linear = eqx.nn.Linear(dims[i], dims[i+1], key=keys[i])
            self.layers.append(linear)
            if i < len(dims) - 2:  # No activation on output
                self.layers.append(jax.nn.swish)
    
    def _forward(self, x: jnp.ndarray) -> jnp.ndarray:
        y = x
        for layer in self.layers:
            y = layer(y)
        return y

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim > 1:
            return jax.vmap(self._forward)(x)
        return self._forward(x)

class BayesDeepONet(eqx.Module):
    """Bayesian DeepONet with uncertainty quantification"""
    branch_net: BranchNet
    trunk_net: TrunkNet
    log_var: jnp.ndarray  # Log variance for uncertainty
    
    def __init__(self, 
                 branch_input_dim: int,
                 trunk_input_dim: int, 
                 hidden_dims: list = [64, 64, 64],
                 latent_dim: int = 64,
                 output_dim: int = 2,
                 key: jax.random.PRNGKey = jax.random.PRNGKey(0)):
        
        key1, key2, key3 = jax.random.split(key, 3)
        
        # Branch network processes input function representations
        self.branch_net = BranchNet(branch_input_dim, hidden_dims, latent_dim, key1)
        
        # Trunk network processes spatial coordinates
        self.trunk_net = TrunkNet(trunk_input_dim, hidden_dims, latent_dim, key2)
        
        # Initialize log variance for uncertainty
        self.log_var = jax.random.normal(key3, (output_dim,)) * 0.1
    
    def __call__(self, branch_input: jnp.ndarray, trunk_input: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            branch_input: [batch, branch_input_dim] - input function encoding
            trunk_input: [..., trunk_input_dim] - spatial coordinates; accepts (batch, num_points, dim) or (num_points, dim)
            
        Returns:
            mean: broadcast to match trunk_input leading dims with final dimension output_dim
            variance: same shape as mean
        """
        # Process through networks
        branch_out = self.branch_net(branch_input)  # [batch, latent_dim]
        if trunk_input.ndim == 2:
            trunk_out = self.trunk_net(trunk_input)
            mean = jnp.sum(branch_out * trunk_out, axis=-1, keepdims=True)
        else:
            batch, num_pts, coord_dim = trunk_input.shape
            trunk_flat = jnp.reshape(trunk_input, (batch * num_pts, coord_dim))
            trunk_proj = self.trunk_net(trunk_flat)
            trunk_proj = jnp.reshape(trunk_proj, (batch, num_pts, -1))
            mean = jnp.sum(trunk_proj * branch_out[:, None, :], axis=-1, keepdims=True)
        variance = jnp.exp(self.log_var)
        variance = jnp.broadcast_to(variance, mean.shape[:-1] + (self.log_var.shape[0],))
        if mean.shape[-1] == 1:
            mean = jnp.broadcast_to(mean, mean.shape[:-1] + (self.log_var.shape[0],))
        return mean, variance
    
    def sample(self, branch_input: jnp.ndarray, trunk_input: jnp.ndarray, key: jax.random.PRNGKey, n_samples: int = 1):
        """Sample from the predictive distribution"""
        mean, variance = self(branch_input, trunk_input)
        samples = mean + jnp.sqrt(variance) * jax.random.normal(key, (n_samples,) + mean.shape)
        return samples
