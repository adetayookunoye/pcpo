"""
Collect timing benchmarks for computational efficiency analysis.

Measures inference time for each model and tracks compute cost vs accuracy.
"""

import argparse
import time
import json
import os
from typing import Dict, List, Tuple
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


def count_parameters(model: eqx.Module) -> int:
    """Count trainable parameters in model."""
    def count(x):
        return jnp.prod(jnp.array(x.shape)) if hasattr(x, 'shape') else 0
    
    params = eqx.filter(model, eqx.is_array)
    return sum(int(count(p)) for p in jax.tree_util.tree_leaves(params))


def benchmark_model(
    model: eqx.Module,
    x: jnp.ndarray,
    model_name: str,
    num_runs: int = 10,
) -> Dict[str, float]:
    """
    Benchmark model inference time.
    
    Args:
        model: Trained model
        x: Input sample
        model_name: Name of model
        num_runs: Number of runs for averaging
        
    Returns:
        Dictionary with timing statistics
    """
    
    # Warm-up
    print("  Warming up...", end="", flush=True)
    for _ in range(3):
        if model_name == "cvae_fno":
            key = jax.random.PRNGKey(0)
            _ = model(x, key=key)
        else:
            _ = model(x)
    print(" ✅")
    
    # Benchmark
    print(f"  Benchmarking ({num_runs} runs)...", end="", flush=True)
    times = []
    
    for i in range(num_runs):
        key = jax.random.PRNGKey(i)
        
        start = time.time()
        if model_name == "cvae_fno":
            _ = model(x, key=key)
        else:
            _ = model(x)
        elapsed = time.time() - start
        
        times.append(elapsed * 1000)  # Convert to ms
    
    print(" ✅")
    
    stats = {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "median_ms": float(np.median(times)),
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Collect timing benchmarks")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--models", type=str, default="fno,divfree_fno,cvae_fno,pino,bayes_deeponet",
                       help="Comma-separated model names")
    parser.add_argument("--num-runs", type=int, default=20, help="Number of timing runs per model")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for benchmarking")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    print("\n" + "="*80)
    print("⏱️  COMPUTATIONAL EFFICIENCY BENCHMARKS")
    print("="*80 + "\n")
    
    # Load sample data
    print("Loading sample data...")
    data_source, manifest = build_data_source(cfg["data"])
    key = jax.random.PRNGKey(42)
    key, data_key = jax.random.split(key)
    x_batch, _ = data_source.sample_batch(data_key, batch_size=args.batch_size)
    
    # Prepare output
    results_dir = cfg["outputs"]["results_dir"]
    timing_dir = os.path.join(results_dir, "timing_benchmarks")
    os.makedirs(timing_dir, exist_ok=True)
    
    models_to_eval = args.models.split(",")
    
    # Collect benchmarks for all models
    all_benchmarks = {}
    
    for model_name in models_to_eval:
        print(f"\n{'='*80}")
        print(f"Model: {model_name.upper()}")
        print(f"{'='*80}")
        
        try:
            # Load model
            key, model_key = jax.random.split(key)
            model = get_model(model_name, cfg, model_key)
            model = AmplitudeWrapper(model, enabled=False)
            
            # Count parameters
            num_params = count_parameters(model)
            print(f"Parameters: {num_params:,}")
            
            # Benchmark
            print(f"Input shape: {x_batch.shape}")
            timing_stats = benchmark_model(model, x_batch[0], model_name, num_runs=args.num_runs)
            
            all_benchmarks[model_name] = {
                "num_parameters": num_params,
                "input_shape": str(x_batch[0].shape),
                "batch_size": args.batch_size,
                "timing_ms": timing_stats,
            }
            
            print(f"\nTiming Statistics:")
            print(f"  Mean: {timing_stats['mean_ms']:.3f} ms")
            print(f"  Std:  {timing_stats['std_ms']:.3f} ms")
            print(f"  Med:  {timing_stats['median_ms']:.3f} ms")
            print(f"  Min:  {timing_stats['min_ms']:.3f} ms")
            print(f"  Max:  {timing_stats['max_ms']:.3f} ms")
            
        except Exception as e:
            print(f"\n❌ Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    output_file = os.path.join(timing_dir, "timing_benchmarks.json")
    save_json(all_benchmarks, output_file)
    print(f"\n✅ Saved timing benchmarks to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: Efficiency Rankings")
    print("="*80)
    
    # Sort by inference time
    sorted_models = sorted(
        all_benchmarks.items(),
        key=lambda x: x[1]["timing_ms"]["mean_ms"]
    )
    
    print("\nInference Time (fastest to slowest):")
    for rank, (model_name, data) in enumerate(sorted_models, 1):
        mean_time = data["timing_ms"]["mean_ms"]
        num_params = data["num_parameters"]
        print(f"  {rank}. {model_name:20s} {mean_time:8.3f} ms ({num_params:,} params)")
    
    # Calculate speedup relative to slowest
    if sorted_models:
        slowest_time = sorted_models[-1][1]["timing_ms"]["mean_ms"]
        fastest_time = sorted_models[0][1]["timing_ms"]["mean_ms"]
        speedup = slowest_time / fastest_time
        
        print(f"\nSpeedup (fastest vs slowest): {speedup:.2f}x")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
