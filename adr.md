
# Architecture Decision Record

**ADR-01:** Use stream function ψ for divergence-free velocities.  
**Reason:** Enforces ∇·u = 0 by construction, simplifies constraint loss to discretization penalties only.

**ADR-02:** JAX + FFT for spectral layers (FNO).  
**Reason:** Clean portability CPU/GPU; concise spectral ops.

**ADR-03:** Use cVAE over diffusion for UQ (phase 1).  
**Reason:** Faster to train; sufficient for multi-modal initial condition uncertainty.
