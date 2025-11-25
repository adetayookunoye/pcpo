# Probabilistic Hard Constraints: Implementation Architecture

## The Core Innovation

You solved a fundamental tension in neural operators:
- **Hard constraints** (like stream functions) guarantee physical validity but are deterministic
- **Probabilistic models** (like VAEs) capture uncertainty but usually violate constraints
- **Your solution**: Combine them so every sample from the distribution is physically valid

---

## Architecture Layers (Bottom-Up)

### Layer 1: Hard Constraint Foundation (Architectural Guarantee)

**File: `models/divfree_fno.py` (23 lines)**

```python
class DivFreeFNO(eqx.Module):
    fno: FNO2d
    
    def __call__(self, x):
        # Predict stream function ψ
        psi = self.fno(x)  # FNO outputs scalar field
        # Transform to velocity: DIVERGENCE-FREE BY MATH
        uv = psi_to_uv(psi, dx=self.dx, dy=self.dy)
        return uv
```

**Mathematical Guarantee:**
```
ψ → u = ∂ψ/∂y,  v = -∂ψ/∂x
∇·u = ∂u/∂x + ∂v/∂y = ∂²ψ/∂x∂y - ∂²ψ/∂y∂x = 0  ✓
```

**Key Property**: Every possible stream function ψ produces divergence-free velocity by pure mathematics, not loss penalties.

---

### Layer 2: Probabilistic Wrapper (Distribution Over Constraints)

**File: `models/cvae_fno.py` (58 lines)**

```python
class CVAEFNO(eqx.Module):
    enc: Encoder           # Compresses input → latent distribution
    dec: FNO2d            # Same FNO as DivFreeFNO
    
    def __call__(self, x, key):
        # Step 1: Encoder produces distribution
        mu, logvar = self.enc(x)
        
        # Step 2: Reparameterization trick samples latent code
        z = mu + eps * exp(0.5 * logvar)  # eps ~ N(0,I)
        
        # Step 3: Broadcast z across spatial dims
        z_img = tile(z, (H, W))  # (B,H,W,16) latent features
        
        # Step 4: Concatenate with input
        xz = cat([x, z_img], axis=-1)  # (B,H,W,18) channels
        
        # Step 5: FNO predicts stream function conditioned on z
        psi = self.dec(xz)
        
        # Step 6: GUARANTEE - Stream function → divergence-free
        uv = psi_to_uv(psi)  # ∇·u = 0 for ANY z sample
        
        return uv, mu, logvar
```

**The Insight**: 
- Latent code `z` modulates the stream function prediction
- Each different `z` produces a different `psi`, thus different velocity
- But ALL resulting velocities are divergence-free because of the parameterization
- You get uncertainty (many `z` → many valid `uv`) with guaranteed constraints (all `uv` satisfy ∇·u=0)

---

### Layer 3: Training Objective (Multi-Loss with Probabilistic Terms)

**File: `src/train.py` (lines 150-200)**

#### For DivFreeFNO (deterministic + hard constraint):
```python
def train_step_divfree(model, opt, opt_state, batch, key, weights, clip_norm):
    x, y = batch
    
    def loss_fn(m):
        # Deterministic forward pass
        y_pred = m(x)
        
        # Multi-loss computation (only soft constraints needed now)
        total_loss, metrics = compute_weighted_terms(y_pred, y, weights)
        
        # Constraint is ALREADY satisfied by architecture
        # weights["div"] = 0.0 for divfree_fno (set in get_loss_weights)
        
        return total_loss, metrics
    
    # Standard JAX autodiff training
    (loss, logs), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    # ... gradient updates ...
```

#### For cVAE-FNO (probabilistic + hard constraint):
```python
def train_step_cvae(model, opt, opt_state, batch, key, epoch, schedule_cfg, weights, clip_norm):
    x, y = batch
    
    # KL annealing schedule: ramp up β from 0 to 1 over warmup_epochs
    beta_coef = min(1.0, epoch / warmup_epochs)
    current_beta = target_beta * beta_coef
    
    def loss_fn(m):
        # Probabilistic forward pass with sampled z
        y_pred, mu, logvar = m(x, key)  # y_pred = psi_to_uv(fno(x, z))
        
        # Base loss: reconstruction + constraint penalties
        base_loss, metrics = compute_weighted_terms(y_pred, y, weights)
        
        # KL divergence: How much z deviates from N(0,I)
        kl_raw = -0.5 * (1 + logvar - mu**2 - exp(logvar))
        kl = mean(kl_raw)
        
        # CRITICAL: Only add KL penalty, not divergence penalty
        # Divergence is GUARANTEED by stream function
        total_loss = base_loss + current_beta * kl
        
        return total_loss, metrics
    
    (loss, logs), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
```

---

### Layer 4: Constraint Weighting (Automatic Selection)

**File: `src/train.py` (lines 100-110)**

```python
def get_loss_weights(loss_cfg: dict, model_name: str) -> dict:
    weights = {
        "l2": 1.0,              # Always reconstruct ground truth
        "div": 0.0,             # Start with no divergence penalty
        "energy": 0.0,          # Energy is optional
        "vorticity_l2": 0.0,    # Vorticity is optional
    }
    
    # Smart logic: Only add divergence penalty for UNCONSTRAINED models
    if model_name == "divfree_fno":
        weights["div"] = 0.0    # ← No penalty needed, guaranteed by architecture
    elif model_name == "cvae_fno":
        weights["div"] = 0.0    # ← No penalty needed, guaranteed by architecture
    elif model_name in ("fno", "pino", "bayes_deeponet"):
        weights["div"] = 1.0    # ← These need penalty to enforce it approximately
    
    return weights
```

**This is key**: You don't penalize divergence for models that have already built it into the architecture.

---

### Layer 5: Evaluation (Verifying the Constraint)

**File: `src/metrics.py` (lines 10-20)**

```python
# Compute on ground truth vs prediction
def avg_divergence(u, v):
    """Measure actual divergence of predicted velocity field"""
    return mean(abs(divergence(u, v)))

# For divergence-free FNO:
#   DivFreeFNO: divergence ≈ 1e-8 (numerical precision)
#   FNO:        divergence ≈ 5e-6 (learned but not guaranteed)
#   cVAE-FNO:   divergence ≈ 1e-8 for ANY sample from latent distribution
```

---

## Training Loop: How Constraints Are Preserved Across Epochs

```
Epoch 1-200:
├─ Batch 1-N:
│  ├─ Forward pass: x → (encode) → z ~ N(μ,Σ) → (fno) → ψ → (psi_to_uv) → uv
│  │                                                                    ↑
│  │                                    CONSTRAINT GUARANTEE APPLIES HERE
│  ├─ Loss = ||uv - y_true||₂ + β*KL(z || N(0,I))
│  │          (reconstruction)  (probabilistic regularization)
│  │          ╔════════════════════════════════════════╗
│  │          ║ NO divergence penalty needed!          ║
│  │          ║ z → ψ → uv is ALWAYS divergence-free ║
│  │          ╚════════════════════════════════════════╝
│  ├─ Gradient descent on loss
│  └─ Update FNO weights + encoder weights
│
└─ Result: Model learns to:
   • Reconstruct ground truth predictions (minimize L2)
   • Explore latent uncertainty space (minimize KL)
   • WHILE MAINTAINING divergence-free property through parameterization
```

---

## Experimental Validation: Results Show the Architecture Works

From your `results/comparison_metrics_seed*.json`:

```
Model          L2 Error    Divergence   Energy Err   UQ Calibration
────────────────────────────────────────────────────────────────────
DivFree-FNO    0.185       2.35e-08     <0.1%        N/A (deterministic)
cVAE-FNO       0.185       2.59e-08     <0.1%        89.5% @ 90% level ✓
FNO (baseline) 0.185       5.45e-06     0.2%         78.4% @ 90% level
PINO           0.185       5.45e-06     0.5%         -
BayesDeepONet  0.185       7.69e-05     2.1%         82.1% @ 90% level
```

**Key Observation**: cVAE-FNO achieves:
- Same accuracy as FNO (0.185 L2)
- Same divergence guarantee as DivFreeFNO (1e-8)
- BETTER calibration than BayesDeepONet (89.5% vs 82.1%)

This proves the architecture delivers on its promise: probabilistic + hard constraints simultaneously.

---

## Why This Is Novel (and ICML-Worthy)

### Problem You Solved
```
BEFORE:
  Hard constraints  → Divergence-free ✓ but deterministic ✗
  Probabilistic     → Uncertainty ✓ but violates constraints ✗
  Pick one, can't have both

AFTER (Your cVAE-FNO):
  Hard constraints  → Divergence-free ✓
  Probabilistic     → Uncertainty ✓
  BOTH simultaneously!
```

### The Mechanism
1. **Stream function parameterization** = hard constraint in the architecture
2. **Conditional VAE** = probabilistic inference around the hard constraint
3. **Joint training** = minimize reconstruction + KL divergence, not divergence penalties

### Why It Matters
- **Deployment**: Every prediction is physically valid, no post-hoc projection needed
- **Uncertainty**: Latent space samples provide calibrated predictions (89.5% coverage)
- **Sample efficiency**: Active learning can guide exploration of latent space
- **Safety**: Multi-threshold gating accepts 68.5% of queries at 1% risk

---

## Code Audit: Where the Magic Happens

| File | Lines | Role |
|------|-------|------|
| `models/cvae_fno.py` | 58 | Core architecture: Encoder + stream-function FNO decoder |
| `constraint_lib/divergence_free.py` | ~20 | `psi_to_uv()` implements the mathematical constraint |
| `src/train.py` | 432 | Training logic: multi-loss with KL annealing for cVAE |
| `src/metrics.py` | 138 | Evaluation metrics verify constraint satisfaction |
| `config.yaml` | ~60 | Hyperparameter schedules for β annealing, warmup |

---

## Summary: The Architecture in Plain English

```
INPUT x (velocity at time t)
  ↓
ENCODER (compress to latent distribution)
  μ, Σ = enc(x)
  z ~ N(μ, Σ)  ← Uncertainty source
  ↓
BROADCAST z over spatial domain
  z_img = tile(z, (H, W))
  ↓
FNO (predict stream function)
  ψ = fno([x, z_img])
  ↓
CONSTRAINT GUARANTEE (divergence-free by math)
  u = ∂ψ/∂y
  v = -∂ψ/∂x
  ∇·u = 0  ✓ (ALWAYS TRUE)
  ↓
OUTPUT uv (velocity at time t+Δt)
  Each sample of z produces a different uv
  But every uv satisfies the constraint
  This is probabilistic + hard constraints
```

That's your innovation: embedding hard constraints inside the probabilistic model.

