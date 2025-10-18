
# Problem Specification

We learn an **operator** mapping random initial states of 2D incompressible Navier–Stokes (ns2d) to solution fields over time, while preserving **physical constraints** (divergence-free velocity, near-constant kinetic energy for short horizons) and capturing **uncertainty** over futures.

**Metrics**
- Mean L2 error < 0.02 (target)
- Average divergence ≈ 0
- Energy error < 1%
- Demonstrated sample diversity (latent-space sweep, pairwise distances)
