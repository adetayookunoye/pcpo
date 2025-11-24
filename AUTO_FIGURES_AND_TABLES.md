# 🤖 Automatic Figure & Table Generation After Training

**Status**: ✅ Complete automatic post-training pipeline implemented

## Overview

The project now includes **automatic figure and table generation** after training and evaluation complete. No manual intervention needed!

## How It Works

### Automatic Trigger Points

#### 1. **After All Evaluation Completes** (Automatic)
When you run `make eval-all`, the final evaluation command automatically triggers the post-training pipeline:

```bash
make eval-all
# ↓ (automatically triggers after all seeds evaluated)
# 🚀 Post-training automation starts:
#    - Aggregates comparison metrics
#    - Generates comparison plots
#    - Generates 20 publication figures
#    - Generates gates analysis
#    - Validates physics
#    - Creates summary report
```

#### 2. **Manual Trigger** (Explicit)
You can manually trigger post-training pipeline anytime:

```bash
make post-training
# or
python -m src.post_training --config config.yaml --results-dir results --figures-dir results/figures
```

#### 3. **Watch Mode** (Monitor & Auto-Trigger)
Continuously monitor for training completion and auto-run when ready:

```bash
python -m src.post_training \
  --watch \
  --watch-seeds 0,1,2,3,4 \
  --watch-interval 60 \
  --config config.yaml \
  --results-dir results \
  --figures-dir results/figures
```

---

## Usage Examples

### Quick Start (Automatic)
```bash
# Train and evaluate everything, figures auto-generate at the end
make compare

# Equivalent to:
make train-all eval-all post-training validate
```

### Full Reproduction (Automatic)
```bash
# Everything including figures/tables
make reproduce-all

# Equivalent to:
make init download train-all eval-all post-training validate zip
```

### Step-by-Step with Auto-Trigger
```bash
# Train all models
make train-all

# Evaluate all models (auto-triggers post-training at the end)
make eval-all

# Results: All 20 figures + tables automatically in results/figures/
```

---

## Pipeline Stages

The post-training automation runs these stages **sequentially**:

```
1. ✅ Comparison Metrics Aggregation
   └─ Combines all seed-wise metrics
   └─ Output: results/compare.md, results/compare.csv

2. ✅ Comparison Plots
   └─ Bar charts, error bars, CIs
   └─ Output: results/figures/{comparison_*.png}

3. ✅ Publication Figures (20 total)
   └─ Fig 1-3: Core comparison
   └─ Fig 4-7: Diagnostics  
   └─ Fig 8-10: Physics validation
   └─ Fig 11: Convergence
   └─ Fig 12-16: Comparative analysis
   └─ Fig 17-20: Extended analysis
   └─ Output: results/figures/{fig1_*.png ... fig20_*.png}

4. ✅ Gates Analysis
   └─ Statistical gating/filtering
   └─ Output: results/gates.csv (if applicable)

5. ✅ Physics Validation (Optional: --skip-physics-validation)
   └─ Validates divergence-free constraints
   └─ Output: results/physics_validation.json

6. ✅ Summary Report
   └─ Comprehensive overview of all generated artifacts
   └─ Output: results/TRAINING_SUMMARY.md
```

---

## Auto-Trigger Implementation Details

### 1. Training Auto-Marker
**File**: `src/train.py` (lines 417-419)

After training each model/seed, a completion marker is written:
```python
complete_marker = os.path.join(results_dir, f".{model}_seed{seed}_complete")
with open(complete_marker, 'w') as f:
    f.write(str(int(time.time())))
```

### 2. Evaluation Auto-Trigger
**File**: `src/eval.py` (lines 293-316)

After all-models evaluation completes, automatically call post-training:
```python
if args.all_models:
    # ... save results ...
    
    # Auto-trigger post-training pipeline
    subprocess.run([
        "python", "-m", "src.post_training",
        "--config", config_file,
        "--results-dir", results_dir,
        "--figures-dir", figures_dir
    ], check=False)
```

### 3. Post-Training Pipeline
**File**: `src/post_training.py`

Comprehensive automation script that:
- Checks for training/eval completion
- Runs all pipeline stages
- Generates summary report
- Supports watch mode

---

## Output Structure

After post-training completes:

```
results/
├── compare.md                          # Markdown table with metrics
├── compare.csv                         # CSV metrics
├── TRAINING_SUMMARY.md                 # Summary report (NEW!)
├── comparison_metrics_seed*.json       # Per-seed metrics
├── {model}_train_history.json          # Training curves
├── {model}_eval_metrics.json           # Per-model evaluation
│
└── figures/
    ├── fig1_model_comparison.png
    ├── fig2_divergence_constraint.png
    ├── fig3_uncertainty_quantification.png
    ├── fig4_rollout_diagnostics.png
    ├── fig5_spectral_analysis.png
    ├── fig6_vorticity_visualization.png
    ├── fig7_seed_stability.png
    ├── fig8_uq_calibration.png
    ├── fig9_energy_conservation.png
    ├── fig10_divergence_spatial_map.png
    ├── fig11_convergence_curves.png
    ├── fig12_radar_plot.png
    ├── fig13_error_histograms.png
    ├── fig14_crps_decomposition.png
    ├── fig15_efficiency_scatter.png
    ├── fig16_phase_space.png
    ├── fig17_ablation_framework.png
    ├── fig18_pde_resolution_heatmap.png
    ├── fig19_ensemble_agreement.png
    └── fig20_multi_pde_summary.png
```

---

## Makefile Targets

### Available Commands

```bash
# Single model/seed training
make train-fno SEED=0
make eval-fno SEED=0

# All models, all seeds
make train-all                  # Train all 5 models × 5 seeds
make eval-all                   # Evaluate all (auto-triggers post-training)

# Manual pipeline control
make post-training              # Manually trigger post-training
make aggregate                  # Manually aggregate metrics
make plots                      # Manually generate plots
make figures                    # Manually generate 20 publication figures
make gates                       # Manually generate gates analysis

# Combined workflows
make compare                    # train-all + eval-all + post-training + gates
make validate                   # Run physics validation
make reproduce-all              # Full pipeline: init download train-all eval-all ...

# Deployment
make deploy                     # Build Docker image
```

---

## Advanced Usage

### Custom Pipelines

#### Run evaluation without post-training
```bash
# Disable auto-trigger by running individual eval
python -m src.eval --config config.yaml --model fno --seed 0
```

#### Run post-training on subset of seeds
```bash
python -m src.post_training \
  --config config.yaml \
  --results-dir results \
  --figures-dir results/figures
  # (reads all comparison_metrics_seed*.json files found)
```

#### Skip specific post-training steps
```bash
python -m src.post_training \
  --config config.yaml \
  --results-dir results \
  --figures-dir results/figures \
  --skip-physics-validation        # Skip physics validation
```

#### Watch for completion (server mode)
```bash
# Terminal 1: Start watching
python -m src.post_training \
  --watch \
  --watch-seeds 0,1,2,3,4 \
  --watch-interval 60               # Check every 60 seconds
  --config config.yaml \
  --results-dir results

# Terminal 2: Run training
make train-all eval-all
# Watch will automatically trigger post-training when ready
```

---

## Configuration

### Default Behavior
- **Automatic**: After `eval-all`, post-training runs automatically
- **Output**: `results/figures/` contains all 20 PNG figures
- **Tables**: `results/compare.{md,csv}` contain metrics
- **Summary**: `results/TRAINING_SUMMARY.md` documents everything

### Customization via Command Line

```bash
# Custom output directories
python -m src.post_training \
  --config my_config.yaml \
  --results-dir /custom/results \
  --figures-dir /custom/figures

# Skip optional steps
python -m src.post_training \
  --skip-physics-validation \
  --results-dir results

# Watch mode with custom interval
python -m src.post_training \
  --watch \
  --watch-interval 120 \           # Check every 2 minutes
  --watch-seeds 0,1,2,3,4
```

---

## Troubleshooting

### Q: Post-training didn't run automatically
**A**: Check that you ran `eval-all` (not individual eval commands)
```bash
# ✅ This triggers auto post-training
make eval-all

# ❌ This does NOT trigger auto post-training  
python -m src.eval --config config.yaml --model fno --seed 0
```

### Q: Want to regenerate figures without re-training
**A**: Use manual post-training trigger
```bash
make post-training
# or
python -m src.post_training --config config.yaml --results-dir results --figures-dir results/figures
```

### Q: How long does post-training take?
**A**: 
- Metrics aggregation: ~10 seconds
- Comparison plots: ~30 seconds
- 20 publication figures: ~2-3 minutes
- Gates analysis: ~10 seconds
- Physics validation: ~30 seconds
- **Total**: ~4-5 minutes

### Q: Can I run training/evaluation in parallel?
**A**: Yes! Each seed is independent:
```bash
# Terminal 1
make train-fno SEED=0

# Terminal 2
make train-fno SEED=1

# Terminal 3
make train-fno SEED=2

# Then evaluate all when done
make eval-all  # Auto-triggers post-training
```

### Q: figures directory already exists, will it overwrite?
**A**: Yes, new figures overwrite old ones in `results/figures/`

### Q: How do I monitor progress?
**A**: 
```bash
# Watch output in real-time
tail -f results/TRAINING_SUMMARY.md

# Check what's been generated
ls -lh results/figures/

# Count figure files
ls results/figures/fig*.png | wc -l
```

---

## Integration with Existing Workflows

### CI/CD Integration
```yaml
# .github/workflows/train-and-generate.yml
name: Train and Generate Figures
on: [push, pull_request]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install -e .
      - name: Download data
        run: make download
      - name: Train and generate figures
        run: make compare          # Auto-triggers post-training
      - name: Upload artifacts
        uses: actions/upload-artifact@v2
        with:
          name: figures
          path: results/figures/
```

### Docker Integration
```dockerfile
# Dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -e .
CMD ["make", "compare"]  # Auto-generates figures when container runs
```

---

## Files Modified

### Core Changes
1. **`src/train.py`**
   - Added time import
   - Added completion marker at end of training

2. **`src/eval.py`**
   - Added time import  
   - Added auto-trigger of post-training after all-models eval

3. **`src/post_training.py`** (NEW)
   - Complete post-training automation pipeline
   - Supports manual trigger and watch mode

4. **`Makefile`**
   - Added `post-training` target
   - Updated `compare` to include post-training
   - Updated `reproduce-all` to include post-training

### Documentation
- `AUTO_FIGURES_AND_TABLES.md` (this file)

---

## Summary

### Before (Manual)
```bash
make train-all
make eval-all
make aggregate
make plots
make figures
make gates
# ~10 manual commands
```

### After (Automatic)
```bash
make compare
# All figures & tables auto-generated! 🎉
```

---

## Quick Reference

| Task | Command | Auto-Triggers Post? |
|------|---------|-------------------|
| Train single | `make train-fno SEED=0` | ❌ No |
| Eval single | `python -m src.eval ...` | ❌ No |
| Train all | `make train-all` | ❌ No |
| Eval all | `make eval-all` | ✅ Yes |
| Full workflow | `make compare` | ✅ Yes |
| Manual post-training | `make post-training` | ✅ Runs directly |
| Watch mode | `python -m src.post_training --watch` | ✅ When ready |
| Full reproduction | `make reproduce-all` | ✅ Yes |

---

**Status**: ✅ IMPLEMENTED - Ready for use!

