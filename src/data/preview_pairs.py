import argparse
from pathlib import Path

import numpy as np

from src.data.pdebench_ns2d import load_pairs_from_npz, _manifest_path  # type: ignore


def print_sample(idx: int, x_sample: np.ndarray, y_sample: np.ndarray):
    """Pretty-print a single velocity pair with basic statistics."""
    def summary(arr: np.ndarray) -> str:
        return (
            f"shape={arr.shape} min={arr.min():.4f} max={arr.max():.4f} "
            f"mean={arr.mean():.4f} std={arr.std():.4f}"
        )

    print(f"\nSample {idx}")
    print(f"  x: {summary(x_sample)}")
    print(f"     first row (center): {x_sample[0, 0]}")
    print(f"  y: {summary(y_sample)}")
    print(f"     first row (center): {y_sample[0, 0]}")


def main():
    parser = argparse.ArgumentParser(
        description="Preview the first N velocity pairs from processed PDEBench data."
    )
    parser.add_argument("--root", type=str, default="./data_cache", help="Data root directory.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ns_incom",
        help="Processed dataset identifier (e.g., ns_incom).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_ds*_stride*_off*.npz",
        help="Glob pattern to locate processed NPZ files.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of samples to preview.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    processed_dir = root / "processed" / args.dataset

    if not processed_dir.exists():
        raise FileNotFoundError(
            f"Processed directory {processed_dir} does not exist. "
            "Run ppo-download first."
        )

    manifest = _manifest_path(root, args.dataset)
    if manifest.exists():
        print(f"Manifest: {manifest.read_text().strip()}")

    dataset = load_pairs_from_npz(processed_dir, pattern=args.pattern)
    x = np.array(dataset.x)
    y = np.array(dataset.y)

    n = min(args.count, x.shape[0])
    print(f"Previewing {n} samples from {processed_dir} ({x.shape[0]} total).")
    for idx in range(n):
        print_sample(idx, x[idx], y[idx])


if __name__ == "__main__":
    main()
