"""
Extract vorticity field predictions for visualization.

Runs inference on trained models and computes vorticity maps (curl of velocity)
for qualitative comparison between predicted and ground truth fields.
"""

import argparse
import json
import os
from typing import Dict, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from src.utils import load_config, save_json
from src.data.pdebench_ns2d import build_data_source
from models.fno import FNO2d
from models.divfree_fno import DivFreeFNO
from models.cvae_fno import CVAEFNO
from models.pino import PINO
from models.bayesian_deeponet import BayesDeepONet
from src.model_wrappers import AmplitudeWrapper


def compute_vorticity(u: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """
    Compute vorticity field: ω = ∂v/∂x - ∂u/∂y
    
    Args:
        u: Velocity component u (H, W)
        v: Velocity component v (H, W)
        
    Returns:
        Vorticity field (H, W)
    """
    # Compute derivatives using finite differences
    dv_dx = (jnp.roll(v, -1, axis=1) - jnp.roll(v, 1, axis=1)) / 2.0
    du_dy = (jnp.roll(u, -1, axis=0) - jnp.roll(u, 1, axis=0)) / 2.0
    
    vorticity = dv_dx - du_dy
    return vorticity


def get_model(model_name: str, cfg: dict, key: jnp.ndarray):
    """Load trained model."""
    models = {
        "fno": FNO2d,
        "divfree_fno": DivFreeFNO,
        "cvae_fno": CVAEFNO,
        "pino": PINO,
        "bayes_deeponet": BayesDeepONet,
    }
    
    model_class = models.get(model_name)
    if model_class is None:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model_class(cfg)
    
    # Load checkpoint
    ckpt_dir = cfg["outputs"]["checkpoints_dir"].format(model=model_name)
    ckpt_path = os.path.join(ckpt_dir, "last_ckpt.npz")
    
    if os.path.exists(ckpt_path):
        model = eqx.tree_deserialise_leaves(ckpt_path, model)
        print(f"✅ Loaded checkpoint: {ckpt_path}")
    else:
        print(f"⚠️ No checkpoint found: {ckpt_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Extract vorticity field predictions")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--models", type=str, default="fno,divfree_fno,cvae_fno,pino,bayes_deeponet",
                       help="Comma-separated model names")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of test samples")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    print("\n" + "="*80)
    print("🌀 EXTRACTING VORTICITY FIELD PREDICTIONS")
    print("="*80 + "\n")
    
    # Load data
    print("Loading data...")
    data_source, manifest = build_data_source(cfg["data"])
    
    # Prepare output
    results_dir = cfg["outputs"]["results_dir"]
    vorticity_dir = os.path.join(results_dir, "vorticity_fields")
    os.makedirs(vorticity_dir, exist_ok=True)
    
    models_to_eval = args.models.split(",")
    key = jax.random.PRNGKey(args.seed)
    
    # Collect vorticity fields for all models and samples
    all_vorticities = {}
    
    for model_name in models_to_eval:
        print(f"\n{'='*80}")
        print(f"Model: {model_name.upper()}")
        print(f"{'='*80}")
        
        try:
            # Load model
            key, model_key = jax.random.split(key)
            model = get_model(model_name, cfg, model_key)
            model = AmplitudeWrapper(model, enabled=False)
            
            model_data = {
                "predictions": [],
                "ground_truth": [],
                "vorticity_pred": [],
                "vorticity_gt": [],
            }
            
            for sample_idx in range(args.num_samples):
                # Get sample
                key, data_key = jax.random.split(key)
                x_batch, y_batch = data_source.sample_batch(data_key, batch_size=1)
                x = x_batch[0]
                y = y_batch[0]  # (timesteps, H, W, 2)
                
                print(f"  Sample {sample_idx+1}/{args.num_samples}...", end="")
                
                try:
                    # Get prediction (use first timestep)
                    if model_name == "cvae_fno":
                        # Probabilistic: average multiple samples
                        key, subkey = jax.random.split(key)
                        preds = []
                        for _ in range(3):
                            key, s = jax.random.split(key)
                            pred = model(x, key=s)
                            preds.append(pred)
                        pred = jnp.mean(jnp.array(preds), axis=0)
                    else:
                        pred = model(x)
                    
                    # Get ground truth (first timestep output)
                    gt = y[0]
                    
                    # Extract velocity components
                    u_pred = np.array(pred[..., 0])
                    v_pred = np.array(pred[..., 1])
                    u_gt = np.array(gt[..., 0])
                    v_gt = np.array(gt[..., 1])
                    
                    # Compute vorticity
                    vort_pred = np.array(compute_vorticity(pred[..., 0], pred[..., 1]))
                    vort_gt = np.array(compute_vorticity(gt[..., 0], gt[..., 1]))
                    
                    # Store (convert to lists for JSON serialization)
                    model_data["predictions"].append(np.concatenate([u_pred[None], v_pred[None]], axis=0).tolist())
                    model_data["ground_truth"].append(np.concatenate([u_gt[None], v_gt[None]], axis=0).tolist())
                    model_data["vorticity_pred"].append(vort_pred.tolist())
                    model_data["vorticity_gt"].append(vort_gt.tolist())
                    
                    print(" ✅")
                    
                except Exception as e:
                    print(f" ❌ ({e})")
            
            all_vorticities[model_name] = model_data
            print(f"\n✅ Completed {model_name}: {len(model_data['predictions'])} samples")
            
        except Exception as e:
            print(f"\n❌ Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    output_file = os.path.join(vorticity_dir, f"vorticity_fields_seed{args.seed}.json")
    # Note: JSON doesn't handle large numpy arrays well, so we'll save summary stats instead
    
    summary = {}
    for model_name, data in all_vorticities.items():
        if data["vorticity_pred"]:
            vort_preds = np.array(data["vorticity_pred"])
            vort_gts = np.array(data["vorticity_gt"])
            
            summary[model_name] = {
                "num_samples": len(data["vorticity_pred"]),
                "vort_pred_mean": float(np.mean(np.abs(vort_preds))),
                "vort_pred_std": float(np.std(vort_preds)),
                "vort_gt_mean": float(np.mean(np.abs(vort_gts))),
                "vort_gt_std": float(np.std(vort_gts)),
                "vort_error": float(np.mean(np.abs(vort_preds - vort_gts))),
            }
    
    save_json(summary, output_file)
    print(f"\n✅ Saved vorticity summary to: {output_file}")
    
    # Also save field data to numpy file for visualization
    np_file = os.path.join(vorticity_dir, f"vorticity_fields_seed{args.seed}.npz")
    np.savez(np_file, **{
        f"{model}_pred": np.array(data["vorticity_pred"])
        for model, data in all_vorticities.items()
    })
    print(f"✅ Saved vorticity fields to: {np_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for model_name, stats in summary.items():
        print(f"\n{model_name}:")
        print(f"  Samples: {stats['num_samples']}")
        print(f"  Vort Pred (μ±σ): {stats['vort_pred_mean']:.4f}±{stats['vort_pred_std']:.4f}")
        print(f"  Vort GT (μ±σ): {stats['vort_gt_mean']:.4f}±{stats['vort_gt_std']:.4f}")
        print(f"  Vort Error: {stats['vort_error']:.4f}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
