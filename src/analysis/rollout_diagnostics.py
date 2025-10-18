import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.utils import load_config
from src.data.synthetic_ns2d import generate_batch
from src.data.pdebench_ns2d import build_data_source
from src.metrics import (
    l2,
    avg_divergence,
    energy_conservation,
    vorticity_l2,
    enstrophy_rel_err,
    spectrum,
    pde_residual_surrogate,
)
from src.eval import load_model


def _coord_grid(grid_size: int) -> jnp.ndarray:
    xs = jnp.linspace(0.0, 1.0, grid_size, dtype=jnp.float32)
    ys = jnp.linspace(0.0, 1.0, grid_size, dtype=jnp.float32)
    xv, yv = jnp.meshgrid(xs, ys, indexing="xy")
    coords = jnp.stack([xv, yv], axis=-1)
    return jnp.reshape(coords, (-1, 2))


def spectral_ratio_curve(u_pred: jnp.ndarray, v_pred: jnp.ndarray, u_true: jnp.ndarray, v_true: jnp.ndarray, num_bins: int = 32):
    pred_spec = spectrum(u_pred) + spectrum(v_pred)
    true_spec = spectrum(u_true) + spectrum(v_true)
    ratio = pred_spec / (true_spec + 1e-8)
    h, w = ratio.shape
    ky = jnp.fft.fftfreq(h)[:, None]
    kx = jnp.fft.fftfreq(w)[None, :]
    radii = jnp.sqrt(kx**2 + ky**2)
    max_radius = float(radii.max())
    bins = jnp.linspace(0.0, max_radius, num_bins + 1)
    curves = []
    for i in range(num_bins):
        mask = (radii >= bins[i]) & (radii < bins[i + 1])
        if jnp.any(mask):
            curves.append(float(jnp.nanmean(ratio[mask])))
        else:
            curves.append(np.nan)
    centers = 0.5 * (np.array(bins[:-1]) + np.array(bins[1:]))
    return centers, np.array(curves)


def rollout(model, model_name, seq_raw, stats, key, samples, coord_flat):
    x_mean, x_std, y_mean, y_std = stats
    B, steps, H, W, C = seq_raw.shape
    assert B == 1, "Rollout diagnostics currently supports batch_size=1."
    energy_ref = seq_raw[:, 0]
    current_raw = seq_raw[:, 0]
    current_norm = (current_raw - x_mean) / x_std
    divs = []
    energy_errs = []
    l2s = []
    spectra = []
    ratios = []

    centers_ref = None
    for t in range(1, steps):
        gt_raw = seq_raw[:, t]
        if model_name == "cvae_fno":
            preds = []
            for _ in range(samples):
                key, subkey = jax.random.split(key)
                pred_norm, _, _ = model(current_norm, subkey)
                preds.append(pred_norm)
            preds_norm = jnp.stack(preds, axis=0)
            pred_raw = preds_norm * y_std + y_mean
            pred_raw_mean = jnp.mean(pred_raw, axis=0)
        elif model_name == "bayes_deeponet":
            branch_input = jnp.reshape(current_norm, (B, H * W * C))
            trunk = jnp.broadcast_to(coord_flat[None, :, :], (B, coord_flat.shape[0], coord_flat.shape[1]))
            mean, _ = model(branch_input, trunk)
            pred_norm = jnp.reshape(mean, (B, H, W, C))
            pred_raw = pred_norm * y_std + y_mean
            pred_raw_mean = pred_raw
        else:
            pred_norm = model(current_norm)
            pred_raw = pred_norm * y_std + y_mean
            pred_raw_mean = pred_raw

        l2s.append(float(l2(pred_raw_mean, gt_raw)))
        divs.append(float(avg_divergence(pred_raw_mean[..., 0], pred_raw_mean[..., 1])))
        energy_errs.append(float(energy_conservation(pred_raw_mean[..., 0], pred_raw_mean[..., 1], gt_raw[..., 0], gt_raw[..., 1])))
        centers, ratio = spectral_ratio_curve(pred_raw_mean[0, ..., 0], pred_raw_mean[0, ..., 1], gt_raw[0, ..., 0], gt_raw[0, ..., 1])
        if centers_ref is None:
            centers_ref = centers
        ratios.append(ratio)
        spectra.append(float(pde_residual_surrogate(pred_raw_mean[..., 0], pred_raw_mean[..., 1])))

        current_raw = pred_raw_mean
        current_norm = (current_raw - x_mean) / x_std

    return {
        "divergence": np.array(divs),
        "energy_rel_err": np.array(energy_errs),
        "l2": np.array(l2s),
        "residual": np.array(spectra),
        "spectral_ratio": np.array(ratios),
        "spectral_centers": centers_ref if centers_ref is not None else np.array([]),
    }


def plot_curves(metrics, out_dir, model_name):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot generation.")
        return
    timesteps = np.arange(1, metrics["divergence"].shape[0] + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(timesteps, metrics["divergence"], label="Divergence")
    plt.plot(timesteps, metrics["energy_rel_err"], label="Energy rel err")
    plt.xlabel("Rollout step")
    plt.ylabel("Metric")
    plt.title(f"{model_name} divergence / energy drift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}_drift.png"))
    plt.close()

    ratio = metrics["spectral_ratio"]
    centers = metrics.get("spectral_centers", np.array([]))
    if ratio.size > 0 and centers.size > 0:
        mean_ratio = np.nanmean(ratio, axis=0)
        plt.figure(figsize=(8, 4))
        plt.plot(centers, mean_ratio, label="E_hat/E")
        plt.axhline(1.0, color="k", linestyle="--")
        plt.xlabel("Frequency bin")
        plt.ylabel("Energy ratio")
        plt.title(f"{model_name} spectral energy ratio")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{model_name}_spectrum_ratio.png"))
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    key = jax.random.PRNGKey(args.seed)

    dataset, _ = build_data_source(cfg["data"])
    x_mean = dataset.x_mean
    x_std = dataset.x_std
    y_mean = dataset.y_mean
    y_std = dataset.y_std

    B = 1
    H = cfg["data"]["grid_size"]
    W = cfg["data"]["grid_size"]

    key, rollout_key = jax.random.split(key)
    _, _, seq_raw = generate_batch(rollout_key, batch_size=B, H=H, W=W, steps=args.steps, return_seq=True)

    stats = (x_mean, x_std, y_mean, y_std)
    model = load_model(args.model, cfg, args.checkpoint)
    coord_flat = _coord_grid(H) if args.model == "bayes_deeponet" else None

    metrics = rollout(model, args.model, seq_raw, stats, key, args.samples, coord_flat)

    out_dir = args.output_dir or os.path.join(cfg["outputs"]["figures_dir"], "diagnostics")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plot_curves(metrics, out_dir, args.model)

    json_path = os.path.join(out_dir, f"{args.model}_rollout_metrics.json")
    def _nan_to_none(obj):
        if isinstance(obj, list):
            return [_nan_to_none(x) for x in obj]
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    serializable = {k: _nan_to_none(np.array(v).tolist()) for k, v in metrics.items()}
    import json
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=2)
    print(f"Saved diagnostics to {json_path} and plots in {out_dir}")


if __name__ == "__main__":
    main()
