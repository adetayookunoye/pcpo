# LaTeX Paper Guide

## Overview

This directory contains a complete AISTAT-formatted paper on Provably Constrained Probabilistic Operators. The paper includes:

- **Competitive abstract** highlighting novelty
- **Complete introduction** with problem motivation
- **Comprehensive literature review** (2020-2025)
- **Mathematical preliminaries** and formal foundations
- **Four technical contributions**:
  1. DivFree-FNO architecture with formal proofs
  2. cVAE-FNO probabilistic extension
  3. Multi-constraint framework (Helmholtz decomposition)
  4. Adaptive constraint weighting
- **Theoretical analysis** with 5 main theorems
- **Comprehensive experiments** (5 seeds, statistical validation)
- **Discussion and implications**
- **Extensive appendices**:
  - Complete proofs (Theorems 1-5)
  - JAX implementation algorithms
  - Ablation studies (4 ablations)
  - Extended related work
  - Code reproduction guide

## Files

```
latex/
├── main.tex              # Main paper (currently AISTAT format)
├── references.bib        # Bibliography with 30+ citations
└── README.md            # This file
```

## Compilation

### Requirements

```bash
# Install LaTeX (Ubuntu/Debian)
sudo apt-get install texlive-full

# Install LaTeX (macOS with Homebrew)
brew install mactex

# Install LaTeX (using MikTeX on Windows)
# Download from https://miktex.org/
```

### Build Commands

```bash
# Basic compilation
pdflatex -interaction=nonstopmode main.tex

# With bibliography
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

# One-command build (Unix/Mac)
latexmk -pdf main.tex

# Or using Makefile
make pdf
```

### Output

After compilation, you'll have:
- `main.pdf` - Full paper (~20 pages)
- `main.aux` - Auxiliary file
- `main.bbl` - Bibliography entries

## Paper Structure

### Main Paper (main.tex)

1. **Title and Abstract** (§0)
   - ~200 word abstract highlighting key contributions
   - Clearly states novelty claims

2. **Introduction** (§1)
   - Problem motivation (divergence violations, lack of UQ)
   - Main contributions (5 listed)
   - Paper organization

3. **Related Work** (§2)
   - Neural operators (2020-2025): FNO, DeepONet, PINO
   - Physics constraints in DL
   - UQ for operators
   - Divergence-free representations
   - Multi-constraint learning
   - **Gap clearly identified**: First systematic application of stream function to spectral operators

4. **Preliminaries** (§3)
   - Neural operators formulation
   - Incompressible flow constraints
   - Stream function definition
   - FNO background
   - Conditional VAE overview

5. **Methods** (§4)
   - **DivFree-FNO** (§4.1)
     - Architecture definition
     - **Theorem 1: Divergence-Free Guarantee** (with proof)
     - Corollary: Discrete error analysis
     - Table: Comparison to penalty methods
   
   - **cVAE-FNO** (§4.2)
     - Architecture definition
     - Proposition: Constrained uncertainty
   
   - **Multi-Constraint Framework** (§4.3)
     - Helmholtz decomposition theorem
     - MultiConstraint-FNO architecture
   
   - **Adaptive Weighting** (§4.4)
     - Learned spatial gating mechanism

6. **Theoretical Analysis** (§5)
   - **Theorem 1**: Hard vs Soft Guarantees (with proof)
   - **Theorem 2**: Universal approximation for divergence-free fields
   - **Theorem 3**: Discretization error bounds
   - All theorems have complete proofs

7. **Experiments** (§6)
   - Dataset: PDEBench 2D NS, 5 seeds
   - Baselines: 6 methods (FNO, FNO+Penalty, PINO, Bayes-DeepONet, DivFree-FNO, cVAE-FNO)
   - Metrics: L2, divergence, energy, vorticity spectrum, UQ metrics
   - Training setup

8. **Results** (§7)
   - **Table 1**: Main results - DivFree-FNO achieves 300× divergence reduction
   - **Table 2**: UQ metrics - cVAE-FNO provides better calibration
   - **Figure/Table**: Multi-constraint decomposition
   - **Table 3**: Timing - negligible overhead
   - All results with 95% confidence intervals

9. **Discussion** (§8)
   - Why architectural constraints work
   - Generalization beyond divergence-free
   - Limitations and future work
   - Practical implications

10. **Conclusion** (§9)
    - Summary of contributions
    - Impact and future directions

## Appendices

### Appendix A: Additional Proofs (§A)
- Extended proof of Theorem 1 (divergence-free guarantee)
- Full proof of Corollary 1 (discrete error)

### Appendix B: Implementation Details (§B)
- Algorithm 1: DivFree-FNO forward pass (JAX pseudocode)
- Algorithm 2: cVAE-FNO training step (JAX pseudocode)

### Appendix C: Ablation Studies (§C)
- **Ablation 1**: Finite difference schemes (forward vs central vs backward)
  - Central differences provide best accuracy
  
- **Ablation 2**: Stream function vs direct prediction
  - Shows penalty methods fail at high λ, stream method robust
  
- **Ablation 3**: VAE β parameter tuning
  - β=1.0 provides optimal balance
  
- **Ablation 4**: Adaptive weighting gate
  - Learned weighting maintains guarantees while learning heterogeneity

### Appendix D: Extended Related Work (§D)
- Hard vs soft constraints discussion
- Conservation laws in ML
- Surrogate modeling trends

### Appendix E: Code and Reproducibility (§E)
- GitHub repository reference
- Installation instructions
- Reproduction scripts

### Appendix F: Novelty Claims (§F)
- Boxed summary of 5 main contributions
- Clear differentiation from literature

## Customization

### To Submit to Different Venues

#### For AISTAT (Current)
Already in AISTAT format. Use `\usepackage[accepted]{aistats2024}`

#### For NeurIPS/ICML
Replace the preamble package line:
```latex
\usepackage[accepted]{aistats2024}
```
with:
```latex
\documentclass{article}
\usepackage{neurips_2024}
```

#### For arXiv
Keep current settings, compile as-is. The PDF will be compatible.

#### For Journal Submission
Modify line `\usepackage[accepted]{aistats2024}` to use journal-specific package (e.g., `siam` for SIAM journals).

### Customizing Author Information

Edit lines ~15-20:
```latex
\author{
  Your Name \\
  Your Department \\
  \texttt{your.email@example.edu}
}
```

### Adding/Removing Sections

- To remove appendices: Delete `\appendix` and everything after
- To add figures: Use `\begin{figure}...\end{figure}` blocks
- To add tables: Use `\begin{table}...\end{table}` blocks

## Key Novelty Statements (Highlighted in Boxes)

The paper contains **two noveltybox statements** that clearly highlight contributions:

1. **First noveltybox** (after introduction motivation):
   > "Our Core Contribution: We show that constraint satisfaction can be moved from the loss function into the architecture itself..."

2. **Second noveltybox** (after UQ motivation):
   > "Our Second Contribution: We extend the stream function architecture to include probabilistic inference..."

3. **Novelty Claims Summary** (Appendix F):
   - 5 contributions clearly boxed
   - Each with specific quantitative claims

## Citation Information

When citing this paper, use:

```bibtex
@article{okunoye2025stream,
  title={Stream Function Neural Operators with Probabilistic Inference: Guaranteed Physical Constraints and Multi-Scale Learning},
  author={Okunoye, Adetayo},
  journal={AISTAT},
  year={2025}
}
```

## Figures and Tables

Current paper includes:
- **6 Tables** with experimental results and comparisons
- **2 Algorithms** for JAX implementation
- **Multiple boxed highlights** for novelty

To add figures:
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.7\textwidth]{path/to/figure.pdf}
\caption{Your caption}
\label{fig:label}
\end{figure}
```

## Word Count

Current paper: ~8,000 words (excluding appendices)
- Abstract: 200 words
- Main paper: 7,200 words
- Appendices: 4,000+ words

## References

35+ citations covering:
- Neural operators (FNO, DeepONet, PINO)
- Physics-informed ML (PINNs)
- Constraint enforcement
- Uncertainty quantification
- Classical fluid mechanics

## Common Issues

### "aistats2024 not found"
```bash
# Download from: https://github.com/aistats/aistats2024
# Or use standard article class instead
```

### Compilation errors with figures
- Ensure PDF paths are correct
- Use `\usepackage{graphicx}` (already included)
- Convert EPS to PDF if needed

### Bibliography not appearing
```bash
# Run full cycle:
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

### Overfull hbox warnings
- Usually harmless for submission
- Can adjust figure sizes or text width if needed

## Next Steps

1. **Customize author information** (line ~17)
2. **Add figures** from your results (analysis/figures/)
3. **Verify all citations** compile without errors
4. **Generate PDF**: `latexmk -pdf main.tex`
5. **Review and edit** for your specific submission
6. **Submit to venue** or arXiv

## Support

For LaTeX help:
- [Overleaf Tutorials](https://www.overleaf.com/learn)
- [LaTeX Stack Exchange](https://tex.stackexchange.com/)
- [AISTAT Submission Guide](https://www.aistats.org/)

For paper-specific questions:
- See YOUR_NOVEL_METHOD_ANALYSIS.md for novelty claims
- See NOVELTY_AND_PROBLEM_STATEMENT.md for problem motivation
