# Full PINO implementation with physics-informed components
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple

from models.fno import FNO2d
from models.bayesian_deeponet import BayesDeepONet

class PINO(eqx.Module):
    """Physics-Informed Neural Operator with FNO backbone and physics constraints"""
    core: FNO2d
    physics_weight: float
    nu: float
    
    def __init__(self, 
                 in_ch: int = 2,
                 out_ch: int = 2, 
                 width: int = 32,
                 modes: int = 12,
                 depth: int = 4,
                 physics_weight: float = 0.1,
                 nu: float = 1e-3,
                 key: jax.random.PRNGKey = jax.random.PRNGKey(0)):
        
        self.core = FNO2d(in_ch=in_ch, out_ch=out_ch, width=width, modes=modes, depth=depth, key=key)
        self.physics_weight = physics_weight
        self.nu = nu
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Standard forward pass"""
        return self.core(x)
    
    def physics_loss(self, x: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
        """
        Compute physics-informed loss components
        Args:
            x: input tensor [batch, height, width, channels]
            y_pred: predicted output [batch, height, width, 2]
        Returns:
            physics_loss: combined physics constraint loss
        """
        batch_size, h, w, _ = y_pred.shape
        
        # 1. Divergence-free constraint (for incompressible flows)
        div_loss = self._divergence_loss(y_pred)
        
        # 2. Boundary condition enforcement
        bc_loss = self._boundary_condition_loss(y_pred)
        
        # 3. Physical conservation laws (simplified)
        conservation_loss = self._conservation_loss(x, y_pred)

        # 4. PDE residual surrogate (Navier–Stokes diffusion term)
        residual_loss = self._pde_residual_loss(y_pred)
        
        total_physics_loss = (
            div_loss + 
            bc_loss + 
            conservation_loss +
            residual_loss
        )
        
        return total_physics_loss
    
    def _divergence_loss(self, u: jnp.ndarray) -> jnp.ndarray:
        """Compute divergence of vector field u = [u_x, u_y]"""
        u_x = u[..., 0]  # x-component
        u_y = u[..., 1]  # y-component
        
        # Compute gradients using finite differences
        dx_u_x = jnp.gradient(u_x, axis=2)  # gradient in x-direction
        dy_u_y = jnp.gradient(u_y, axis=1)  # gradient in y-direction
        
        divergence = dx_u_x + dy_u_y
        div_loss = jnp.mean(divergence ** 2)
        
        return div_loss
    
    def _boundary_condition_loss(self, u: jnp.ndarray) -> jnp.ndarray:
        """Enforce boundary conditions (no-slip or periodic)"""
        batch_size, h, w, _ = u.shape
        
        # No-slip boundary condition (zero velocity at boundaries)
        top_bc = jnp.mean(u[:, 0, :, :] ** 2)    # top boundary
        bottom_bc = jnp.mean(u[:, -1, :, :] ** 2) # bottom boundary  
        left_bc = jnp.mean(u[:, :, 0, :] ** 2)   # left boundary
        right_bc = jnp.mean(u[:, :, -1, :] ** 2) # right boundary
        
        bc_loss = (top_bc + bottom_bc + left_bc + right_bc) / 4.0
        return bc_loss
    
    def _conservation_loss(self, x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        """Simple energy-like conservation constraint"""
        kinetic_energy = 0.5 * jnp.sum(u ** 2, axis=-1)  # (B,H,W)
        grad_y, grad_x = jnp.gradient(kinetic_energy, axis=(1, 2))
        smoothness_loss = jnp.mean(grad_x ** 2 + grad_y ** 2)
        return smoothness_loss

    def _laplacian(self, f: jnp.ndarray) -> jnp.ndarray:
        return (-4.0 * f +
                jnp.roll(f, 1, axis=-1) + jnp.roll(f, -1, axis=-1) +
                jnp.roll(f, 1, axis=-2) + jnp.roll(f, -1, axis=-2))

    def _pde_residual_loss(self, u: jnp.ndarray) -> jnp.ndarray:
        """Surrogate Navier–Stokes residual using diffusion term only."""
        lap_u = self._laplacian(u[..., 0])
        lap_v = self._laplacian(u[..., 1])
        residual = self.nu * (jnp.abs(lap_u) + jnp.abs(lap_v))
        return jnp.mean(residual)
    
    def training_step(self, x: jnp.ndarray, y_true: jnp.ndarray, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, dict]:
        """Complete training step with data loss + physics loss"""
        # Standard prediction
        y_pred = self(x)
        
        # Data fidelity loss (MSE)
        data_loss = jnp.mean((y_pred - y_true) ** 2)
        
        # Physics regularization
        physics_loss = self.physics_loss(x, y_pred)
        
        # Combined loss
        total_loss = data_loss + self.physics_weight * physics_loss
        
        metrics = {
            'total_loss': total_loss,
            'data_loss': data_loss, 
            'physics_loss': physics_loss,
            'divergence': self._divergence_loss(y_pred),
            'pde_residual': self._pde_residual_loss(y_pred)
        }
        
        return total_loss, metrics

# Example usage and integration with your training pipeline
def create_bayes_deeponet(config: dict, key: jax.random.PRNGKey):
    """Create Bayesian DeepONet from configuration"""
    return BayesDeepONet(
        branch_input_dim=config.get('branch_input_dim', 128),
        trunk_input_dim=config.get('trunk_input_dim', 2),  # (x,y) coordinates
        hidden_dims=config.get('hidden_dims', [64, 64, 64]),
        latent_dim=config.get('latent_dim', 64),
        output_dim=config.get('output_dim', 2),
        key=key
    )

def create_pino(config: dict, key: jax.random.PRNGKey):
    """Create PINO from configuration"""
    return PINO(
        in_ch=config.get('in_channels', 2),
        out_ch=config.get('out_channels', 2),
        width=config.get('width', 32),
        modes=config.get('modes', 12),
        depth=config.get('depth', 4),
        physics_weight=config.get('physics_weight', 0.1),
        nu=config.get('nu', 1e-3),
        key=key
    )
