from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from src.data.synthetic_ns2d import generate_batch as generate_synthetic_batch


@dataclass
class NSPairsDataset:
    """Thin wrapper around downsampled PDEBench Navier–Stokes training pairs with normalization."""

    x: jnp.ndarray
    y: jnp.ndarray
    x_mean: jnp.ndarray
    x_std: jnp.ndarray
    y_mean: jnp.ndarray
    y_std: jnp.ndarray
    normalize_inputs: bool = True
    normalize_targets: bool = True
    augment_std: float = 0.0
    eps: float = 1e-6

    @property
    def size(self) -> int:
        return int(self.x.shape[0])

    def normalize_input(self, x: jnp.ndarray) -> jnp.ndarray:
        return (x - self.x_mean) / self.x_std

    def normalize_output(self, y: jnp.ndarray) -> jnp.ndarray:
        return (y - self.y_mean) / self.y_std

    def denormalize_input(self, x_norm: jnp.ndarray) -> jnp.ndarray:
        return x_norm * self.x_std + self.x_mean

    def denormalize_output(self, y_norm: jnp.ndarray) -> jnp.ndarray:
        return y_norm * self.y_std + self.y_mean

    def sample_batch(
        self,
        key: jax.Array,
        batch_size: int,
        normalize: bool | None = None,
        augment_std: float | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if normalize is None:
            normalize_in = self.normalize_inputs
            normalize_out = self.normalize_targets
        else:
            normalize_in = normalize_out = normalize

        if augment_std is None:
            augment_std = self.augment_std

        key_indices, key_noise = jax.random.split(key)
        indices = jax.random.randint(key_indices, (batch_size,), minval=0, maxval=self.size)
        x_batch = self.x[indices]
        y_batch = self.y[indices]

        if augment_std and augment_std > 0.0:
            noise = augment_std * jax.random.normal(key_noise, x_batch.shape)
            x_batch = x_batch + noise

        if normalize_in:
            x_batch = self.normalize_input(x_batch)
        if normalize_out:
            y_batch = self.normalize_output(y_batch)
        return x_batch, y_batch


class SyntheticFallbackDataset:
    """Wraps the synthetic generator to present the same interface as NSPairsDataset."""

    def __init__(
        self,
        grid_size: int,
        t_steps: int,
        normalize_inputs: bool = True,
        normalize_targets: bool = True,
        augment_std: float = 0.0,
    ):
        self._H = grid_size
        self._W = grid_size
        self._steps = t_steps
        self._stats_initialized = False
        self.normalize_inputs = normalize_inputs
        self.normalize_targets = normalize_targets
        self.augment_std = augment_std
        self.x_mean = jnp.zeros((1, 1, 1, 2), dtype=jnp.float32)
        self.x_std = jnp.ones((1, 1, 1, 2), dtype=jnp.float32)
        self.y_mean = jnp.zeros((1, 1, 1, 2), dtype=jnp.float32)
        self.y_std = jnp.ones((1, 1, 1, 2), dtype=jnp.float32)

    def _maybe_update_stats(self, x: jnp.ndarray, y: jnp.ndarray):
        if not self._stats_initialized:
            self.x_mean = jnp.mean(x, axis=(0, 1, 2), keepdims=True)
            self.x_std = jnp.sqrt(jnp.var(x, axis=(0, 1, 2), keepdims=True) + 1e-6)
            self.y_mean = jnp.mean(y, axis=(0, 1, 2), keepdims=True)
            self.y_std = jnp.sqrt(jnp.var(y, axis=(0, 1, 2), keepdims=True) + 1e-6)
            if not self.normalize_targets:
                self.y_mean = self.x_mean
                self.y_std = self.x_std
            self._stats_initialized = True

    def normalize_input(self, x: jnp.ndarray) -> jnp.ndarray:
        return (x - self.x_mean) / self.x_std

    def normalize_output(self, y: jnp.ndarray) -> jnp.ndarray:
        return (y - self.y_mean) / self.y_std

    def denormalize_input(self, x_norm: jnp.ndarray) -> jnp.ndarray:
        return x_norm * self.x_std + self.x_mean

    def denormalize_output(self, y_norm: jnp.ndarray) -> jnp.ndarray:
        return y_norm * self.y_std + self.y_mean

    def sample_batch(
        self,
        key: jax.Array,
        batch_size: int,
        normalize: bool | None = None,
        augment_std: float | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        normalize_in = self.normalize_inputs if normalize is None else normalize
        normalize_out = self.normalize_targets if normalize is None else normalize
        if augment_std is None:
            augment_std = self.augment_std

        key_gen, key_noise = jax.random.split(key)
        x, y = generate_synthetic_batch(
            key_gen, batch_size=batch_size, H=self._H, W=self._W, steps=self._steps
        )
        self._maybe_update_stats(x, y)

        if augment_std and augment_std > 0.0:
            noise = augment_std * jax.random.normal(key_noise, x.shape)
            x = x + noise

        if normalize_in:
            x = self.normalize_input(x)
        if normalize_out:
            y = self.normalize_output(y)
        return x, y


def load_pairs_from_npz(
    processed_dir: Path,
    pattern: str = "*_ds*_stride*_off*.npz",
    normalize_inputs: bool = True,
    normalize_targets: bool = True,
    augment_std: float = 0.0,
) -> NSPairsDataset:
    files = sorted(processed_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No processed Navier–Stokes files matching pattern '{pattern}' in {processed_dir}"
        )

    x_arrays = []
    y_arrays = []
    for path in files:
        with np.load(path) as data:
            x_arrays.append(np.array(data["x"], dtype=np.float32))
            y_arrays.append(np.array(data["y"], dtype=np.float32))

    x = jnp.asarray(np.concatenate(x_arrays, axis=0))
    y = jnp.asarray(np.concatenate(y_arrays, axis=0))
    x_mean = jnp.mean(x, axis=(0, 1, 2), keepdims=True)
    x_std = jnp.sqrt(jnp.var(x, axis=(0, 1, 2), keepdims=True) + 1e-6)
    y_mean = jnp.mean(y, axis=(0, 1, 2), keepdims=True)
    y_std = jnp.sqrt(jnp.var(y, axis=(0, 1, 2), keepdims=True) + 1e-6)
    if not normalize_targets:
        y_mean = x_mean
        y_std = x_std
    return NSPairsDataset(
        x=x,
        y=y,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        normalize_inputs=normalize_inputs,
        normalize_targets=normalize_targets,
        augment_std=augment_std,
    )


def _manifest_path(root: Path, dataset: str) -> Path:
    return root / "processed" / dataset / "manifest.json"


def read_manifest(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_data_source(
    data_cfg: dict,
) -> tuple[NSPairsDataset | SyntheticFallbackDataset, dict | None]:
    """
    Build the training data source based on processed PDEBench files.

    Returns:
        tuple(dataset, manifest) where manifest is None when synthetic data is used.
    """

    dataset_name = data_cfg.get("dataset", "pdebench_ns2d")
    root = Path(data_cfg.get("root", "./data_cache")).resolve()

    if dataset_name != "pdebench_ns2d":
        raise ValueError(f"Unsupported dataset type '{dataset_name}'.")

    dataset_id = data_cfg.get("pdebench_id", "ns_incom")
    processed_dir = root / "processed" / dataset_id
    pattern = data_cfg.get("processed_pattern", f"{dataset_id}*_ds*_stride*_off*.npz")
    manifest = read_manifest(_manifest_path(root, dataset_id))

    normalize_inputs = bool(data_cfg.get("normalize", True))
    normalize_targets = bool(data_cfg.get("normalize_target_separately", True))
    augment_std = float(data_cfg.get("augment_ic_perturb", 0.0))

    try:
        dataset = load_pairs_from_npz(
            processed_dir,
            pattern=pattern,
            normalize_inputs=normalize_inputs,
            normalize_targets=normalize_targets,
            augment_std=augment_std,
        )
        expected_grid = int(data_cfg.get("grid_size", dataset.x.shape[1]))
        if dataset.x.shape[1] != expected_grid or dataset.x.shape[2] != expected_grid:
            raise ValueError(
                f"Processed data grid size {dataset.x.shape[1]} does not match config grid_size {expected_grid}."
            )
        return dataset, manifest
    except FileNotFoundError:
        if data_cfg.get("synthetic_if_missing", True):
            synthetic = SyntheticFallbackDataset(
                grid_size=data_cfg.get("grid_size", 64),
                t_steps=data_cfg.get("t_steps", 5),
                normalize_inputs=normalize_inputs,
                normalize_targets=normalize_targets,
                augment_std=augment_std,
            )
            return synthetic, None
        raise
