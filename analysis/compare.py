import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

METRICS = [
    "l2",
    "div",
    "energy_err",
    "vorticity_l2",
    "enstrophy_rel_err",
    "spectra_dist",
    "pde_residual",
    "coverage_90",
    "sharpness",
    "crps",
    "mean_speed_pred",
    "mean_speed_true",
    "spectra_ratio_mean",
    "spectra_ratio_std",
]


def bootstrap_ci(xs: np.ndarray, B: int = 1000, alpha: float = 0.05):
    if len(xs) == 0:
        return np.nan, ""
    boots = [np.mean(np.random.choice(xs, size=len(xs), replace=True)) for _ in range(B)]
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    mean = float(np.mean(xs))
    return mean, f"[{lo:.4g}, {hi:.4g}]"


def format_value(val):
    if isinstance(val, float):
        if np.isnan(val):
            return ""
        return f"{val:.6g}"
    return str(val)


def write_markdown(rows, headers, path, inputs):
    md = ["# Model comparison (mean ± 95% CI)", "", f"Inputs: {', '.join(inputs)}", ""]
    md.append("|" + "|".join(headers) + "|")
    md.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        md.append("|" + "|".join(format_value(row.get(h, "")) for h in headers) + "|")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(md))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    per_model = defaultdict(lambda: defaultdict(list))

    for path in args.inputs:
        seed_str = path.split("seed")[-1].split(".")[0]
        try:
            seed = int(seed_str)
        except ValueError:
            seed = 0
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for model, metrics in data.items():
            for metric in METRICS:
                value = metrics.get(metric)
                if isinstance(value, (int, float)):
                    per_model[model][metric].append(float(value))

    rows = []
    for model, metric_dict in per_model.items():
        row = {"model": model}
        for metric in METRICS:
            xs = np.array(metric_dict.get(metric, []), dtype=float)
            mean, ci = bootstrap_ci(xs, B=args.bootstrap)
            row[metric] = mean
            row[f"{metric}_ci"] = ci
        rows.append(row)

    # compute ranks (lower is better)
    for metric in METRICS:
        valid = [(row["model"], row[metric]) for row in rows if not np.isnan(row.get(metric, np.nan))]
        valid.sort(key=lambda x: x[1])
        for rank, (model, _) in enumerate(valid, start=1):
            for row in rows:
                if row["model"] == model:
                    row[f"{metric}_rank"] = rank
                    break

    for row in rows:
        ranks = [row.get(f"{metric}_rank") for metric in METRICS if f"{metric}_rank" in row]
        if ranks:
            row["avg_rank"] = sum(ranks) / len(ranks)
        else:
            row["avg_rank"] = np.nan

    rows.sort(key=lambda r: (float(r["avg_rank"]) if not np.isnan(r["avg_rank"]) else float("inf")))

    # write CSV
    fieldnames = ["model"]
    for metric in METRICS:
        fieldnames.extend([metric, f"{metric}_ci", f"{metric}_rank"])
    fieldnames.append("avg_rank")

    csv_dir = os.path.dirname(args.csv)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out_row = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, float) and np.isnan(value):
                    value = ""
                out_row[key] = value
            writer.writerow(out_row)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_markdown(rows, fieldnames, args.out, args.inputs)
    print(f"Wrote {args.out} and {args.csv}")


if __name__ == "__main__":
    main()
