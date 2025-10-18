
# Technical Report

## Summary
We implement div-free FNO and a cVAE-FNO for 2D incompressible Navier–Stokes. We evaluate physical and statistical metrics and demonstrate uncertainty-aware predictions.

## Methods
- DivFree-FNO: ψ prediction, spectral convolution blocks
- cVAE-FNO: convolutional encoder, FNO decoder

## Results
See `results/` for JSON metrics and plots (vorticity maps, spectra).

## Conclusion
Divergence-free parameterization stabilizes training and improves physical validity. The cVAE offers calibrated spread.
