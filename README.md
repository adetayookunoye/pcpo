
# Provably Constrained Probabilistic Operator Learning (JAX)

This repository implements a research-grade pipeline for learning **physically consistent** surrogate operators on PDEBench's 2D incompressible Navier–Stokes (ns2d).

**Highlights**
- JAX-based **divergence-free FNO** core: model outputs a stream function ψ; velocities are derived as u = ∂ψ/∂y, v = −∂ψ/∂x ⇒ ∇·u = 0 analytically (up to discretization).
- Probabilistic extension: **cVAE** with FNO decoder for uncertainty quantification.
- Physical metrics: L2, divergence, **energy conservation**, **vorticity spectrum**, and **sample diversity**.
- Dual env: **Conda (CPU)** and **Docker (GPU)**.
- One-command reproduction: `make reproduce-all`.

## Quickstart

```bash
# CPU (conda)
conda env create -f environments/conda-cpu.yaml
conda activate ppojax
pip install -e .

# Download + preprocess PDEBench Navier–Stokes data (downsampled to 64×64)
# Example below keeps a single shard to ~9 GB raw / ~200 MB processed.
ppo-download --root ./data_cache --dataset ns_incom --shards 512-0 --max-files 1 --pairs-per-file 1024

# Train a small run (replace model with fno, divfree_fno, cvae_fno, pino, or bayes_deeponet)
ppo-train --config config.yaml --model divfree_fno --epochs 1 --quickrun

# Evaluate (uses the same processed dataset)
ppo-eval --config config.yaml --model divfree_fno --checkpoint results/divfree_fno/checkpoints/last_ckpt.npz

# Validate physics & package
make validate
make zip  # produces final_solution.zip
```

Default settings in `config.yaml` run 200 epochs with a cosine learning-rate decay and per-channel data normalization; tweak the `train.*` entries if you need shorter smoke tests.

Rollout diagnostics (divergence/energy drift curves and spectral energy ratios) can be generated on synthetic sequences via:

```bash
python -m src.analysis.rollout_diagnostics --config config.yaml --model divfree_fno --checkpoint results/divfree_fno/checkpoints/last_ckpt.npz --steps 8
```

## Batch Comparison

Run the full multi-seed benchmark (training, evaluation, aggregation, plots, and gates) with:

```bash
make compare
```

Artifacts:

- `results/comparison_metrics_seed*.json` – per-seed metrics across all models
- `results/compare.csv` / `results/compare.md` – aggregated leaderboard with bootstrap CIs
- `results/figures/*.png` – bar charts for key metrics
- `analysis/gates.py` enforces divergence and coverage tolerances; the `make gates` step fails if they are violated

Override the default settings when experimenting, e.g. `make train-all EPOCHS=1` or `make eval-all N_SAMPLES=32` for quicker sanity checks.

See `problem_spec.md`, `literature_review.md`, `adr.md`, `theoretical_foundations.md`, `report.md`, and `validation.md` for design rationale and results.

## PDEBench Data Download

`ppo-download` now mirrors the official PDEBench dataset registry. By default it
pulls the incompressible Navier–Stokes shards (`ns_incom`) from the DaRUS data
repository, verifies the MD5 checksum, and converts them into downsampled
64×64 velocity pairs stored under `data_cache/processed/ns_incom/`.

Key flags:

- `--shards 512-0 512-1` – choose specific shard IDs (filenames contain the token).
- `--max-files N` – cap how many shards to pull; each 512 shard is ≈9 GB raw.
- `--pairs-per-file M` – limit how many (input,target) examples are kept per shard.
- `--frame-stride S` / `--target-offset O` – control temporal spacing between frames.
- `--delete-raw` – drop the heavy HDF5 files after conversion if space is tight.

Whenever the processed data are absent the training and evaluation scripts fall
back to the synthetic generator so development remains frictionless.
