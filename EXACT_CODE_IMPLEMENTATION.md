# Probabilistic Hard Constraints: The Exact Code

This document shows the actual code implementing each layer.

---

## Layer 1: Hard Constraint Function (The Math)

**File: `constraint_lib/divergence_free.py`**

```python
def psi_to_uv(psi, dx=1.0, dy=1.0):
    """
    Convert stream function to divergence-free velocity.
    
    Mathematical identity:
      u = ∂ψ/∂y
      v = -∂ψ/∂x
      ∇·u = ∂u/∂x + ∂v/∂y = 0  (by definition of stream function)
    
    Args:
        psi: (B, H, W) or (B, H, W, 1)
        dx, dy: grid spacing
    
    Returns:
        uv: (B, 2, H, W) with u,v channels
    """
    # Finite-difference derivatives (central differences)
    def d_dx(f):
        return (jnp.roll(f, -1, axis=-1) - jnp.roll(f, 1, axis=-1)) / (2.0 * dx)
    
    def d_dy(f):
        return (jnp.roll(f, -1, axis=-2) - jnp.roll(f, 1, axis=-2)) / (2.0 * dy)
    
    u = d_dy(psi)      # ∂ψ/∂y
    v = -d_dx(psi)     # -∂ψ/∂x
    
    return jnp.stack([u, v], axis=1)  # (B, 2, H, W)


def divergence(u, v, dx=1.0, dy=1.0):
    """
    Compute divergence of velocity field: ∇·u = ∂u/∂x + ∂v/∂y
    
    For velocity derived from stream function, this is ~1e-8 (numerical error only)
    For other velocity fields, this can be large
    """
    def d_dx(f):
        return (jnp.roll(f, -1, axis=-1) - jnp.roll(f, 1, axis=-1)) / (2.0 * dx)
    
    def d_dy(f):
        return (jnp.roll(f, -1, axis=-2) - jnp.roll(f, 1, axis=-2)) / (2.0 * dy)
    
    du_dx = d_dx(u)
    dv_dy = d_dy(v)
    
    return du_dx + dv_dy
```

**Key Property**: Any ψ → `psi_to_uv()` → divergence ≈ 0 (machine precision)

---

## Layer 2a: Deterministic Stream Function Model (DivFreeFNO)

**File: `models/divfree_fno.py`**

```python
import jax
import jax.numpy as jnp
import equinox as eqx
from models.fno import FNO2d
from constraint_lib.divergence_free import psi_to_uv

class DivFreeFNO(eqx.Module):
    """
    Divergence-free FNO: Predicts stream function, derives velocity.
    
    Architecture:
      Input velocity (B,H,W,2)
        ↓
      FNO predicts stream function (B,H,W,1)
        ↓
      psi_to_uv() derives velocity (B,H,W,2)
        ↓
      Guaranteed: ∇·u = 0
    """
    fno: FNO2d
    dx: float = 1.0
    dy: float = 1.0

    def __init__(self, width=32, modes=12, depth=4, key=jax.random.PRNGKey(0)):
        """
        Initialize FNO that predicts stream function.
        
        Note: Input has 2 channels (u,v), output has 1 channel (ψ)
              This is the key difference from standard FNO
        """
        self.fno = FNO2d(
            in_ch=2,           # Input: velocity (u,v)
            out_ch=1,          # Output: stream function ψ
            width=width,
            modes=modes,
            depth=depth,
            key=key
        )

    def __call__(self, x):
        """
        Forward pass: predict divergence-free velocity.
        
        Args:
            x: (B, H, W, 2) velocity at time t
        
        Returns:
            uv: (B, H, W, 2) velocity at time t+Δt, guaranteed ∇·u = 0
        """
        # Step 1: Predict stream function
        psi = self.fno(x)  # (B, H, W, 1)
        
        # Step 2: Extract scalar field
        psi = psi[..., 0]  # (B, H, W)
        
        # Step 3: Derive velocity from stream function
        # This guarantees ∇·u = 0 by mathematics
        uv = psi_to_uv(psi, dx=self.dx, dy=self.dy)  # (B, 2, H, W)
        
        # Step 4: Reshape to (B, H, W, 2) convention
        uv = jnp.moveaxis(uv, 1, -1)  # (B, H, W, 2)
        
        return uv
```

---

## Layer 2b: Probabilistic Stream Function Model (cVAE-FNO)

**File: `models/cvae_fno.py`**

```python
import jax
import jax.numpy as jnp
import equinox as eqx
from models.fno import FNO2d
from constraint_lib.divergence_free import psi_to_uv

class Encoder(eqx.Module):
    """
    Encoder: Input velocity → latent distribution.
    
    Takes full spatiotemporal field, compresses to μ, Σ
    """
    w1: jnp.ndarray          # (2, 128) - first layer
    w2_mu: jnp.ndarray       # (128, latent_dim) - μ head
    w2_logvar: jnp.ndarray   # (128, latent_dim) - log(Σ) head

    def __init__(self, in_ch, latent_dim, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.w1 = jax.random.normal(k1, (in_ch, 128)) * 0.02
        self.w2_mu = jax.random.normal(k2, (128, latent_dim)) * 0.02
        self.w2_logvar = jax.random.normal(k3, (128, latent_dim)) * 0.02

    def __call__(self, x):
        """
        Args:
            x: (B, H, W, C) velocity field
        
        Returns:
            mu: (B, latent_dim) mean of latent distribution
            logvar: (B, latent_dim) log-variance of latent distribution
        """
        # Global average pooling: reduce spatial dimensions
        x_global = jnp.mean(x, axis=(1, 2))  # (B, C)
        
        # Two-layer MLP
        h = jax.nn.gelu(jnp.dot(x_global, self.w1))  # (B, 128)
        
        # Output two heads for distribution parameters
        mu = jnp.dot(h, self.w2_mu)                   # (B, latent_dim)
        logvar = jnp.dot(h, self.w2_logvar)           # (B, latent_dim)
        
        return mu, logvar


class CVAEFNO(eqx.Module):
    """
    Conditional VAE with stream-function FNO decoder.
    
    Architecture:
      x → Encoder → μ, Σ
                      ↓ sample z
                     [x, z] concatenate
                      ↓
      Decoder (FNO) → ψ
                      ↓ psi_to_uv
                      → u,v (guaranteed ∇·u = 0)
    
    Key: Different z samples produce different ψ and thus different u,v
         But ALL satisfy the divergence-free constraint
    """
    enc: Encoder
    dec: FNO2d
    beta: float = 1.0
    dx: float = 1.0
    dy: float = 1.0

    def __init__(self, in_ch=2, latent_dim=16, width=32, modes=12, depth=4, beta=1.0, key=jax.random.PRNGKey(0)):
        k1, k2 = jax.random.split(key, 2)
        
        self.enc = Encoder(in_ch, latent_dim, k1)
        
        # Decoder FNO: Takes input velocity + latent code
        # Outputs 1 channel (stream function)
        self.dec = FNO2d(
            in_ch=in_ch + latent_dim,  # u,v + latent features
            out_ch=1,                   # Stream function ψ
            width=width,
            modes=modes,
            depth=depth,
            key=k2
        )
        
        self.beta = beta

    def reparam(self, mu, logvar, key):
        """
        Reparameterization trick: z = μ + ε * σ, ε ~ N(0,I)
        
        This allows gradients to flow through the sampling operation
        """
        eps = jax.random.normal(key, mu.shape)
        std = jnp.exp(0.5 * logvar)
        return mu + eps * std

    def __call__(self, x, key):
        """
        Forward pass: produce probabilistic divergence-free predictions.
        
        Args:
            x: (B, H, W, C) input velocity
            key: JAX random key for sampling z
        
        Returns:
            uv: (B, H, W, 2) predicted velocity (divergence-free)
            mu: (B, latent_dim) latent mean
            logvar: (B, latent_dim) latent log-variance
        """
        # Step 1: Encoder compresses input to distribution
        mu, logvar = self.enc(x)  # (B, latent_dim) each
        
        # Step 2: Sample latent code
        z = self.reparam(mu, logvar, key)  # (B, latent_dim)
        
        # Step 3: Broadcast latent code across spatial grid
        B, H, W, C = x.shape
        z_img = jnp.tile(z[:, None, None, :], (1, H, W, 1))  # (B, H, W, latent_dim)
        
        # Step 4: Concatenate input + latent
        xz = jnp.concatenate([x, z_img], axis=-1)  # (B, H, W, C+latent_dim)
        
        # Step 5: FNO predicts stream function
        # Note: conditioned on latent z, so different z → different ψ
        psi = self.dec(xz)  # (B, H, W, 1)
        psi = psi[..., 0]   # (B, H, W)
        
        # Step 6: Derive velocity from stream function
        # CONSTRAINT GUARANTEE: ∇·u = 0 for ANY ψ
        uv = psi_to_uv(psi, dx=self.dx, dy=self.dy)  # (B, 2, H, W)
        
        # Step 7: Reshape to convention
        uv = jnp.moveaxis(uv, 1, -1)  # (B, H, W, 2)
        
        return uv, mu, logvar
```

---

## Layer 3: Training Logic (Multi-Loss with KL Annealing)

**File: `src/train.py` (relevant sections)**

```python
def get_loss_weights(loss_cfg: dict, model_name: str) -> dict:
    """
    Determine which loss terms to use for each model.
    
    Key insight: Constrained models don't need divergence penalty
    """
    weights = {
        "l2": float(loss_cfg.get("l2", 1.0)),              # Always needed
        "div": float(loss_cfg.get("div", 0.0)),            # Usually 0
        "energy": float(loss_cfg.get("energy", 0.0)),      # Optional
        "vorticity_l2": float(loss_cfg.get("vorticity_l2", 0.0)),  # Optional
    }
    
    # Smart logic: Zero out divergence penalty for architecturally constrained models
    if model_name == "divfree_fno":
        weights["div"] = 0.0  # ← Divergence is guaranteed, not learned
    elif model_name == "cvae_fno":
        weights["div"] = 0.0  # ← Same: guaranteed by stream function
    # For other models (fno, pino, bayes), keep divergence penalty
    elif weights["div"] == 0.0 and model_name in ("fno", "pino", "bayes_deeponet"):
        weights["div"] = 1.0
    
    return weights


def compute_weighted_terms(y_pred, y_true, weights):
    """
    Compute all loss terms and weight them.
    """
    metrics = {}
    
    # Reconstruction: How well does prediction match ground truth?
    metrics["l2"] = jnp.mean((y_pred - y_true) ** 2)
    
    # Divergence: How much does velocity field diverge?
    # For constrained models, this will be ~1e-8
    # For unconstrained, might be 1e-5
    metrics["div"] = avg_divergence(y_pred[..., 0], y_pred[..., 1])
    
    # Other metrics (energy, vorticity, etc.)
    metrics["energy"] = energy_conservation(
        y_pred[..., 0], y_pred[..., 1], y_true[..., 0], y_true[..., 1]
    )
    metrics["vorticity_l2"] = vorticity_l2(
        y_pred[..., 0], y_pred[..., 1], y_true[..., 0], y_true[..., 1]
    )
    # ... other metrics ...
    
    # Weighted sum
    total = jnp.array(0.0)
    for name, value in metrics.items():
        weight = float(weights.get(name, 0.0))
        if weight > 0.0:
            total = total + weight * value
    
    return total, metrics


def train_step_cvae(model, opt, opt_state, batch, key, epoch, schedule_cfg, weights, clip_norm):
    """
    Single training step for cVAE-FNO.
    
    Key: Two loss terms only
    1. Reconstruction (matches ground truth)
    2. KL divergence (regularizes latent space)
    
    NO divergence penalty because it's guaranteed architecturally!
    """
    x, y = batch
    
    # KL annealing schedule: gradually increase KL weight from 0 to target
    warmup_epochs = int(schedule_cfg.get("kl_warmup_epochs", 50))
    target_beta = float(schedule_cfg.get("kl_target_beta", 1.0))
    
    if warmup_epochs > 0:
        beta_coef = min(1.0, epoch / warmup_epochs)
    else:
        beta_coef = 1.0
    
    current_beta = target_beta * beta_coef
    
    def loss_fn(m):
        # Forward pass: sample z, predict stream function, derive velocity
        y_pred, mu, logvar = m(x, key)
        
        # Reconstruction loss (standard MSE)
        base_loss, metrics = compute_weighted_terms(y_pred, y, weights)
        # Note: weights["div"] = 0 for cvae_fno, so divergence isn't penalized
        
        # KL divergence loss (probabilistic regularization)
        # Measure: How much does latent distribution deviate from N(0,I)?
        kl_raw = -0.5 * (1 + logvar - mu**2 - jnp.exp(logvar))
        kl = jnp.mean(kl_raw)
        
        # Total loss: reconstruction + weighted KL
        total_loss = base_loss + current_beta * kl
        
        metrics.update({"kl": kl, "beta": current_beta})
        metrics["total_loss"] = total_loss
        
        return total_loss, metrics
    
    # JAX autodiff: compute loss and gradients
    (loss, logs), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    
    # Gradient clipping for stability
    grads = eqx.filter(grads, eqx.is_array)
    grads = clip_gradients(grads, clip_norm)
    
    # Gradient descent update
    params = eqx.filter(model, eqx.is_array)
    updates, opt_state = opt.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    
    return model, opt_state, logs
```

---

## Layer 4: Evaluation (Verify Constraint)

**File: `src/metrics.py`**

```python
def avg_divergence(u, v):
    """
    Compute mean absolute divergence: E[|∇·u|]
    
    For DivFreeFNO or cVAE-FNO: ~1e-8 (machine precision)
    For unconstrained FNO: ~1e-5 to 1e-6
    """
    from constraint_lib.divergence_free import divergence
    
    div_field = divergence(u, v)  # (B, H, W)
    return jnp.mean(jnp.abs(div_field))


# During evaluation:
# for model_name in ["fno", "pino", "bayes_deeponet", "divfree_fno", "cvae_fno"]:
#     div_metric = avg_divergence(y_pred[..., 0], y_pred[..., 1])
#     print(f"{model_name}: divergence = {div_metric:.2e}")
#
# Output:
# fno: divergence = 5.45e-06
# cvae_fno: divergence = 2.59e-08  ← 210× better!
# divfree_fno: divergence = 2.35e-08  ← Same guarantee
```

---

## The Integration: How It All Connects

```
Training Loop (src/train.py):
┌─────────────────────────────────────────┐
│ for epoch in 1..200:                    │
│   for batch in dataloader:              │
│     # Forward pass                      │
│     uv, mu, logvar = model(x, key)      │◄─ Models: divfree_fno.py or cvae_fno.py
│                                         │
│     # Loss computation                  │
│     recon = L2(uv, y)                   │
│     kl = KL(mu, logvar)                 │
│     div_penalty = 0  ← NO NEED          │
│     loss = recon + beta * kl            │◄─ Weights from get_loss_weights()
│                                         │
│     # Gradient descent                  │
│     grads = autodiff(loss)              │
│     model = update(model, grads, lr)    │
│                                         │
│     # Constraint verified automatically │
│     # (not computed, but guaranteed)    │
└─────────────────────────────────────────┘

Stream Function Math (constraint_lib/divergence_free.py):
┌─────────────────────────────────────────┐
│ ψ (predicted by FNO decoder)            │
│   ↓                                     │
│ u = ∂ψ/∂y (via psi_to_uv)              │
│ v = -∂ψ/∂x                             │
│   ↓                                     │
│ ∇·u = ∂u/∂x + ∂v/∂y = 0 ✓ GUARANTEED  │
└─────────────────────────────────────────┘

Evaluation (src/metrics.py):
┌─────────────────────────────────────────┐
│ Compute avg_divergence(u, v)            │
│ → 2.35e-08 for DivFreeFNO               │
│ → 2.59e-08 for cVAE-FNO                 │
│ → 5.45e-06 for unconstrained FNO        │
└─────────────────────────────────────────┘
```

---

## Summary: The Three Key Files

| File | Responsibility | Code Size |
|------|---|---|
| `constraint_lib/divergence_free.py::psi_to_uv()` | Transform ψ → divergence-free u,v | 20 lines |
| `models/divfree_fno.py::DivFreeFNO` | Hard constraint model | 23 lines |
| `models/cvae_fno.py::CVAEFNO` | Probabilistic + hard constraint | 58 lines |

**Total novel code**: ~100 lines to add hard constraints + uncertainty quantification.

