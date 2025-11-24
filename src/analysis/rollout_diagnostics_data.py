"""
Generate rollout diagnostics data for temporal validation.

This script runs inference on trained models, collecting metrics over multiple
timesteps to visualize error growth and constraint violation evolution.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from src.utils import load_config, save_json
from src.data.pdebench_ns2d import build_data_source
from src.metrics import (
    l2, avg_divergence, energy_conservation, 
    vorticity_l2, enstrophy_rel_err, spectrum, spectra_distance
)
from models.fno import FNO2d
from models.divfree_fno import DivFreeFNO
from models.cvae_fno import CVAEFNO
from models.pino import PINO
from models.bayesian_deeponet import BayesDeepONet
from src.model_wrappers import AmplitudeWrapper


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


def run_rollout(
    model: eqx.Module,
    x_init: jnp.ndarray,
    y_true: jnp.ndarray,
    model_name: str,
    num_steps: int = 5,
) -> Dict[str, List[float]]:
    """
    Run multi-step rollout and collect metrics.
    
    Args:
        model: Trained neural operator
        x_init: Initial condition
        y_true: Ground truth for validation
        model_name: Name of model
        num_steps: Number of rollout steps
        
    Returns:
        Dictionary with metrics over time
    """
    
    metrics_over_time = {
        "l2": [],
        "divergence": [],
        "energy_err": [],
        "vorticity_l2": [],
        "enstrophy_rel_err": [],
    }
    
    # Start with initial condition
    x_current = x_init
    
    for step in range(num_steps):
        try:
            # Get prediction
            if model_name == "cvae_fno" and hasattr(model, "num_samples"):
                # Probabilistic model: sample
                key = jax.random.PRNGKey(step)
                preds = []
                for _ in range(5):  # 5 samples
                    key, subkey = jax.random.split(key)
                    pred = model(x_current, key=subkey)
                    preds.append(pred)
                pred = jnp.mean(jnp.array(preds), axis=0)
            else:
                # Deterministic model
                pred = model(x_current)
            
            # Get ground truth for this step
            if step < len(y_true):
                gt = y_true[step]
            else:
                gt = y_true[-1]  # Use last timestep
            
            # Compute metrics
            u_pred = pred[..., 0]
            v_pred = pred[..., 1]
            u_true = gt[..., 0]
            v_true = gt[..., 1]
            
            l2_err = float(l2(pred, gt))
            div_err = float(avg_divergence(u_pred, v_pred))
            energy_err = float(energy_conservation(pred, gt))
            vort_err = float(vorticity_l2(u_pred, v_pred, u_true, v_true))
            enstr_err = float(enstrophy_rel_err(u_pred, v_pred, u_true, v_true))
            
            metrics_over_time["l2"].append(l2_err)
            metrics_over_time["divergence"].append(div_err)
            metrics_over_time["energy_err"].append(energy_err)
            metrics_over_time["vorticity_l2"].append(vort_err)
            metrics_over_time["enstrophy_rel_err"].append(enstr_err)
            
            print(f"  Step {step+1}: L2={l2_err:.4f}, Div={div_err:.2e}, Energy={energy_err:.4f}")
            
            # Use prediction as input for next step
            x_current = pred
            
        except Exception as e:
            print(f"  ⚠️ Step {step+1} failed: {e}")
            # Use ground truth to continue
            if step < len(y_true) - 1:
                x_current = y_true[step + 1]
    
    return metrics_over_time


def main():
    parser = argparse.ArgumentParser(description="Generate rollout diagnostics")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--models", type=str, default="fno,divfree_fno,cvae_fno,pino,bayes_deeponet",
                       help="Comma-separated model names")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num-steps", type=int, default=5, help="Number of rollout steps")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of test samples")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    print("\n" + "="*80)
    print("🚀 GENERATING ROLLOUT DIAGNOSTICS DATA")
    print("="*80 + "\n")
    
    # Load data
    print("Loading data...")
    data_source, manifest = build_data_source(cfg["data"])
    
    # Prepare output
    results_dir = cfg["outputs"]["results_dir"]
    rollout_dir = os.path.join(results_dir, "rollout_diagnostics")
    os.makedirs(rollout_dir, exist_ok=True)
    
    # Collect data for all models
    all_rollouts = {}
    
    models_to_eval = args.models.split(",")
    key = jax.random.PRNGKey(args.seed)
    
    for model_name in models_to_eval:
        print(f"\n{'='*80}")
        print(f"Model: {model_name.upper()}")
        print(f"{'='*80}")
        
        try:
            # Load model
            key, model_key = jax.random.split(key)
            model = get_model(model_name, cfg, model_key)
            model = AmplitudeWrapper(model, enabled=False)  # Load without amplitude wrapper
            
            # Collect rollouts
            model_rollouts = []
            
            for sample_idx in range(args.num_samples):
                # Get sample
                key, data_key = jax.random.split(key)
                x_batch, y_batch = data_source.sample_batch(data_key, batch_size=1)
                x = x_batch[0]
                y = y_batch[0]  # (timesteps, H, W, 2)
                
                print(f"\n  Sample {sample_idx+1}/{args.num_samples}:")
                rollout_metrics = run_rollout(model, x, y, model_name, num_steps=args.num_steps)
                model_rollouts.append(rollout_metrics)
            
            all_rollouts[model_name] = model_rollouts
            print(f"\n✅ Completed {model_name}")
            
        except Exception as e:
            print(f"\n❌ Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    output_file = os.path.join(rollout_dir, f"rollout_metrics_seed{args.seed}.json")
    save_json(all_rollouts, output_file)
    print(f"\n✅ Saved rollout diagnostics to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for model_name in all_rollouts:
        rollouts = all_rollouts[model_name]
        if rollouts and rollouts[0]:
            first_l2 = np.mean([r["l2"][0] for r in rollouts if r["l2"]])
            final_l2 = np.mean([r["l2"][-1] for r in rollouts if r["l2"]])
            print(f"\n{model_name}:")
            print(f"  Samples: {len(rollouts)}")
            print(f"  Initial L2: {first_l2:.4f}")
            print(f"  Final L2 (step {args.num_steps}): {final_l2:.4f}")
            print(f"  Error growth: {(final_l2/first_l2 - 1)*100:.1f}%")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
