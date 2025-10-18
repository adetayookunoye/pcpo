import argparse
import json
import os
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from src.utils import load_config, save_json
from src.data.pdebench_ns2d import build_data_source
from src.data.synthetic_ns2d import generate_batch
from src.metrics import (
    l2,
    avg_divergence,
    energy_conservation,
    vorticity_l2,
    enstrophy_rel_err,
    spectra_distance,
    spectrum,
    pde_residual_surrogate,
    sample_aggregate,
    sharpness,
    empirical_coverage,
    crps_samples,
)
from models.fno import FNO2d
from models.divfree_fno import DivFreeFNO
from models.cvae_fno import CVAEFNO
from models.pino import PINO
from models.bayesian_deeponet import BayesDeepONet
from src.model_wrappers import AmplitudeWrapper


def _coord_grid(grid_size: int) -> jnp.ndarray:
    xs = jnp.linspace(0.0, 1.0, grid_size, dtype=jnp.float32)
    ys = jnp.linspace(0.0, 1.0, grid_size, dtype=jnp.float32)
    xv, yv = jnp.meshgrid(xs, ys, indexing="xy")
    coords = jnp.stack([xv, yv], axis=-1)
    return jnp.reshape(coords, (-1, 2))


def _deterministic_metrics(pred: jnp.ndarray, target: jnp.ndarray) -> Dict[str, float]:
    u_pred = pred[..., 0]
    v_pred = pred[..., 1]
    u_true = target[..., 0]
    v_true = target[..., 1]
    return {
        "l2": float(l2(pred, target)),
        "div": float(avg_divergence(u_pred, v_pred)),
        "energy_err": float(energy_conservation(u_pred, v_pred, u_true, v_true)),
        "vorticity_l2": float(vorticity_l2(u_pred, v_pred, u_true, v_true)),
        "enstrophy_rel_err": float(enstrophy_rel_err(u_pred, v_pred, u_true, v_true)),
        "spectra_dist": float(spectra_distance(u_pred, v_pred, u_true, v_true)),
        "pde_residual": float(pde_residual_surrogate(u_pred, v_pred)),
    }


def load_model(name: str, cfg: dict, ckpt: str):
    key = jax.random.PRNGKey(cfg.get("seed", 42))
    depth = cfg["model"].get("depth", 4)
    pino_cfg = cfg.get("pino", {})
    if name == "fno":
        model = FNO2d(
            in_ch=2,
            out_ch=2,
            width=cfg["model"]["width"],
            modes=cfg["model"]["modes"],
            depth=depth,
            key=key,
        )
    elif name == "divfree_fno":
        model = DivFreeFNO(
            width=cfg["model"]["width"],
            modes=cfg["model"]["modes"],
            depth=depth,
            key=key,
        )
    elif name == "cvae_fno":
        model = CVAEFNO(
            in_ch=2,
            latent_dim=cfg["model"]["latent_dim"],
            width=cfg["model"]["width"],
            modes=cfg["model"]["modes"],
            depth=depth,
            beta=cfg["model"]["cvae_beta"],
            key=key,
        )
    elif name == "pino":
        model = PINO(
            in_ch=2,
            out_ch=2,
            width=cfg["model"]["width"],
            modes=cfg["model"]["modes"],
            depth=depth,
            physics_weight=cfg["model"].get("physics_weight", 0.1),
            nu=cfg["model"].get("nu", 1e-3),
            key=key,
        )
    elif name == "bayes_deeponet":
        grid = cfg["data"]["grid_size"]
        branch_dim = grid * grid * 2
        latent = cfg["model"].get("latent_dim", 64)
        hidden = cfg["model"].get("deep_hidden", [64, 64, 64])
        model = BayesDeepONet(
            branch_input_dim=branch_dim,
            trunk_input_dim=2,
            hidden_dims=hidden,
            latent_dim=latent,
            output_dim=2,
            key=key,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    amp_cfg = cfg["model"]
    model = AmplitudeWrapper(
        model,
        init_gain=float(amp_cfg.get("amplitude_init", 1.0)),
        reg_weight=float(amp_cfg.get("amplitude_reg", 1e-6)),
        enabled=bool(amp_cfg.get("amplitude_calibration", True)),
    )
    eqx.tree_deserialise_leaves(ckpt, model)
    return model


def default_checkpoint(cfg: dict, model_name: str) -> str:
    ckpt_dir = cfg["outputs"]["checkpoints_dir"].format(model=model_name)
    return os.path.join(ckpt_dir, "last_ckpt.npz")


def evaluate_model(
    model_name: str,
    cfg: dict,
    checkpoint_path: str,
    x_norm: jnp.ndarray,
    y_norm: jnp.ndarray,
    x_raw: jnp.ndarray,
    y_raw: jnp.ndarray,
    n_samples: int,
    rng_key: jax.Array,
    dataset,
    metric_cfg: dict,
) -> Dict[str, float]:
    model = load_model(model_name, cfg, checkpoint_path)

    if model_name == "cvae_fno":
        def _pred_fn(data, sample_key):
            preds, _, _ = model(data, sample_key)
            return preds

        sample_count = max(n_samples, 64)
        Ys_norm = sample_aggregate(_pred_fn, x_norm, rng_key, n_samples=sample_count)
        Ys_raw = dataset.denormalize_output(Ys_norm)
        y_pred = jnp.mean(Ys_raw, axis=0)
        metrics = _deterministic_metrics(y_pred, y_raw)
        augment_metrics(metrics, y_pred, y_raw, metric_cfg)
        metrics.update(
            {
                "coverage_90": float(empirical_coverage(Ys_raw, y_raw, alpha=0.1)),
                "sharpness": float(sharpness(Ys_raw)),
                "crps": float(crps_samples(Ys_raw, y_raw)),
            }
        )
        return metrics

    if model_name == "bayes_deeponet":
        B, H_dim, W_dim, C = x_norm.shape
        coord_flat = _coord_grid(H_dim)
        branch_input = jnp.reshape(x_norm, (B, H_dim * W_dim * C))
        trunk = jnp.broadcast_to(
            coord_flat[None, :, :],
            (branch_input.shape[0], coord_flat.shape[0], coord_flat.shape[1]),
        )
        mean, _ = model(branch_input, trunk)
        y_pred_norm = jnp.reshape(mean, y_norm.shape)
        y_pred = dataset.denormalize_output(y_pred_norm)
        metrics = _deterministic_metrics(y_pred, y_raw)
        augment_metrics(metrics, y_pred, y_raw, metric_cfg)
        return metrics

    y_pred_norm = model(x_norm)
    y_pred = dataset.denormalize_output(y_pred_norm)
    metrics = _deterministic_metrics(y_pred, y_raw)
    augment_metrics(metrics, y_pred, y_raw, metric_cfg)
    return metrics


def augment_metrics(metrics: Dict[str, float], pred: jnp.ndarray, target: jnp.ndarray, metric_cfg: dict) -> None:
    report_mean_speed = bool(metric_cfg.get("report_mean_speed", False))
    report_spectra = bool(metric_cfg.get("spectra_ratio", False))
    if report_mean_speed:
        speed_pred = jnp.sqrt(pred[..., 0] ** 2 + pred[..., 1] ** 2)
        speed_true = jnp.sqrt(target[..., 0] ** 2 + target[..., 1] ** 2)
        metrics["mean_speed_pred"] = float(jnp.mean(speed_pred))
        metrics["mean_speed_true"] = float(jnp.mean(speed_true))
    if report_spectra:
        spec_pred = spectrum(pred[..., 0]) + spectrum(pred[..., 1])
        spec_true = spectrum(target[..., 0]) + spectrum(target[..., 1])
        ratio = spec_pred / (spec_true + 1e-8)
        metrics["spectra_ratio_mean"] = float(jnp.mean(ratio))
        metrics["spectra_ratio_std"] = float(jnp.std(ratio))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, default="divfree_fno")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples for cVAE evaluation")
    parser.add_argument("--all-models", action="store_true", help="Evaluate all models listed in config.compare.models")
    parser.add_argument("--output", type=str, default=None, help="Path for aggregated JSON when --all-models is used")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    compare_cfg = cfg.get("compare", {})
    models_to_eval: List[str]
    if args.all_models:
        models_to_eval = compare_cfg.get(
            "models", ["fno", "pino", "bayes_deeponet", "divfree_fno", "cvae_fno"]
        )
    else:
        models_to_eval = [args.model]

    n_samples = args.n_samples
    if n_samples is None:
        n_samples = int(compare_cfg.get("n_samples", 16))

    batch_size = cfg["train"].get("batch_size", 4)

    dataset, manifest = build_data_source(cfg["data"])
    if manifest:
        print(
            f"Evaluating on PDEBench data: {manifest['total_pairs']} pairs "
            f"from {len(manifest['processed_files'])} files."
        )
    else:
        print("Evaluating on synthetic Navier–Stokes data.")

    key = jax.random.PRNGKey(cfg.get("seed", 42))
    key, data_key = jax.random.split(key)
    x_norm, y_norm = dataset.sample_batch(data_key, batch_size=batch_size, augment_std=0.0)
    x_raw = dataset.denormalize_input(x_norm)
    y_raw = dataset.denormalize_output(y_norm)

    aggregated_results = {}
    for model_name in models_to_eval:
        key, model_key = jax.random.split(key)
        checkpoint = args.checkpoint
        if checkpoint is None or args.all_models:
            checkpoint = default_checkpoint(cfg, model_name)
        metrics = evaluate_model(
            model_name,
            cfg,
            checkpoint,
            x_norm,
            y_norm,
            x_raw,
            y_raw,
            n_samples,
            model_key,
            dataset,
            cfg.get("metrics", {}),
        )
        aggregated_results[model_name] = metrics
        per_model_path = os.path.join(cfg["outputs"]["results_dir"], f"{model_name}_eval_metrics.json")
        save_json(metrics, per_model_path)
        if not args.all_models:
            print(json.dumps(metrics, indent=2))

    if args.all_models:
        output_path = args.output or os.path.join(cfg["outputs"]["results_dir"], "comparison_metrics.json")
        save_json(aggregated_results, output_path)
        print(json.dumps(aggregated_results, indent=2))


if __name__ == "__main__":
    main()
