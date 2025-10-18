import argparse
import csv
import os

PLOT_METRICS = [
    "l2",
    "div",
    "energy_err",
    "vorticity_l2",
    "enstrophy_rel_err",
    "spectra_dist",
    "pde_residual",
]
UQ_METRICS = ["coverage_90", "sharpness", "crps"]


def load_table(path: str):
    with open(path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    return rows


def bar_plot(rows, metric, outdir, log=False):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot generation.")
        return

    models = []
    values = []
    for row in rows:
        value = row.get(metric)
        if value is None or value == "":
            continue
        try:
            value = float(value)
        except ValueError:
            continue
        models.append(row.get("model", ""))
        values.append(value)
    if not values:
        return

    plt.figure()
    plt.bar(models, values)
    if log:
        plt.yscale("log")
    plt.title(metric)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"{metric}.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    rows = load_table(args.csv)
    for metric in PLOT_METRICS:
        bar_plot(rows, metric, args.outdir, log=(metric == "div"))
    for metric in UQ_METRICS:
        bar_plot(rows, metric, args.outdir, log=False)
    print(f"Saved figures to {args.outdir}")


if __name__ == "__main__":
    main()
