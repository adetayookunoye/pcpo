# 🚀 Quick Start: Automatic Figures After Training

## One-Command Everything

```bash
# Train everything, evaluate everything, generate all figures and tables automatically
make compare
```

That's it! After this completes, you'll have:
- ✅ 20 publication-quality figures in `results/figures/`
- ✅ Comparison metrics in `results/compare.{md,csv}`
- ✅ Summary report in `results/TRAINING_SUMMARY.md`

---

## Timeline

```
Your command: make compare
                ↓
         make train-all (5 models × 5 seeds ≈ few hours)
                ↓
         make eval-all (evaluate all seeds)
                ↓
         🤖 POST-TRAINING AUTOMATION STARTS AUTOMATICALLY
                ↓
         ✅ Figures generated (20 total, ~4-5 minutes)
         ✅ Tables aggregated
         ✅ Report created
                ↓
         ALL DONE! 🎉
```

---

## What Gets Generated

### 📊 Figures (20 total)
```
results/figures/
├── fig1_model_comparison.png           (282 KB)
├── fig2_divergence_constraint.png      (357 KB)
├── fig3_uncertainty_quantification.png (477 KB)
├── ... (20 figures total)
└── fig20_multi_pde_summary.png         (282 KB)
```

### 📋 Tables & Reports
```
results/
├── compare.md                    (Markdown table)
├── compare.csv                   (CSV data)
└── TRAINING_SUMMARY.md          (Overview report)
```

---

## Manual Generation (If Needed)

If you already have results but need to regenerate figures:

```bash
# Regenerate all figures from existing results
make post-training

# Or manually
python -m src.post_training \
  --config config.yaml \
  --results-dir results \
  --figures-dir results/figures
```

---

## Advanced: Watch Mode

Run this in one terminal to monitor for completion:

```bash
python -m src.post_training --watch
```

Then run training in another terminal:

```bash
make train-all eval-all
```

The watch mode will automatically trigger post-training when evaluation completes!

---

## What's New?

Before: Manual 10-step process
```bash
make train-all
make eval-all
make aggregate
make plots
make figures
make gates
# ... repeat ...
```

After: One command!
```bash
make compare
```

✨ **All figures and tables auto-generate at the end!** ✨

---

See `AUTO_FIGURES_AND_TABLES.md` for detailed documentation.

