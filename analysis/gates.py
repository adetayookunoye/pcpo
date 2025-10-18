import argparse
import csv
import sys

import numpy as np


def load_table(path: str):
    with open(path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    return rows


def metric_value(rows, model: str, metric: str):
    for row in rows:
        if row.get("model") == model:
            value = row.get(metric)
            if value in (None, ""):
                return np.nan
            try:
                return float(value)
            except ValueError:
                return np.nan
    return np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--div_tol", type=float, default=1e-10)
    parser.add_argument("--coverage_low", type=float, default=0.85)
    parser.add_argument("--coverage_high", type=float, default=0.95)
    args = parser.parse_args()

    rows = load_table(args.csv)

    failures = []

    div_val = metric_value(rows, "divfree_fno", "div")
    if np.isnan(div_val):
        failures.append("divfree_fno divergence metric missing")
    elif div_val > args.div_tol:
        failures.append(f"divfree_fno divergence {div_val} > {args.div_tol}")

    coverage_val = metric_value(rows, "cvae_fno", "coverage_90")
    if np.isnan(coverage_val):
        failures.append("cvae_fno coverage_90 metric missing")
    elif not (args.coverage_low <= coverage_val <= args.coverage_high):
        failures.append(
            f"cvae_fno coverage_90 {coverage_val} outside [{args.coverage_low}, {args.coverage_high}]"
        )

    if failures:
        print("Gate checks failed:")
        for msg in failures:
            print(f" - {msg}")
        sys.exit(1)

    print("All gates passed")


if __name__ == "__main__":
    main()
