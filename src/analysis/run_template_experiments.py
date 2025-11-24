#!/usr/bin/env python3
"""
Master script to run all template figure experiments.

Executes data collection for:
- Fig 4: Rollout diagnostics
- Fig 6: Vorticity visualization
- Fig 15: Efficiency benchmarks
- Fig 16: Phase space plots
- Fig 17: Ablation studies (optional - requires re-training)
- Fig 18 & 20: Multi-PDE evaluation (optional - requires re-training)
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n{'='*80}")
    print(f"▶ {description}")
    print(f"{'='*80}\n")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=str(Path.cwd()))
    
    if result.returncode == 0:
        print(f"\n✅ {description} completed successfully")
        return True
    else:
        print(f"\n❌ {description} failed with code {result.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run template figure generation experiments"
    )
    parser.add_argument(
        "--skip-rollout",
        action="store_true",
        help="Skip rollout diagnostics"
    )
    parser.add_argument(
        "--skip-vorticity",
        action="store_true",
        help="Skip vorticity extraction"
    )
    parser.add_argument(
        "--skip-timing",
        action="store_true",
        help="Skip timing benchmarks"
    )
    parser.add_argument(
        "--skip-ablations",
        action="store_true",
        help="Skip ablation studies (time-consuming)"
    )
    parser.add_argument(
        "--skip-multipde",
        action="store_true",
        help="Skip multi-PDE training (time-consuming)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of seeds to run (for rollout/vorticity)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: only run quick experiments"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🚀 TEMPLATE FIGURE GENERATION EXPERIMENTS")
    print("="*80)
    
    print("\n📊 Running Data Collection for Template Figures...\n")
    
    completed = []
    failed = []
    
    # Quick experiments (< 30 min each)
    quick_experiments = [
        (
            not args.skip_timing,
            ["python", "-m", "src.analysis.benchmark_timing", "--config", args.config],
            "Timing Benchmarks (Fig 15)"
        ),
        (
            not args.skip_vorticity,
            ["python", "-m", "src.analysis.extract_vorticity_fields", "--config", args.config, "--num-samples", "5"],
            "Vorticity Fields (Fig 6)"
        ),
    ]
    
    # Longer experiments
    longer_experiments = [
        (
            not args.skip_rollout,
            ["python", "-m", "src.analysis.rollout_diagnostics_data", "--config", args.config, "--num-samples", "10"],
            "Rollout Diagnostics (Fig 4)"
        ),
    ]
    
    # Very long experiments (several hours)
    expensive_experiments = [
        (
            not args.skip_ablations,
            ["python", "-m", "src.analysis.run_ablations", "--config", args.config],
            "Ablation Studies (Fig 17)"
        ),
        (
            not args.skip_multipde,
            ["python", "-m", "src.analysis.train_multipde", "--config", args.config],
            "Multi-PDE Training (Fig 18, 20)"
        ),
    ]
    
    # Determine which experiments to run
    all_experiments = []
    
    if args.quick:
        # Quick mode: only timing and vorticity
        all_experiments = quick_experiments
        print("Quick mode: Running only quick experiments (timing, vorticity)")
    else:
        # Run all quick + longer experiments
        all_experiments = quick_experiments + longer_experiments
        
        # Ask about expensive experiments
        if not (args.skip_ablations and args.skip_multipde):
            print("\nLong-running experiments available:")
            print("  - Ablation Studies (Fig 17): ~4-6 hours")
            print("  - Multi-PDE Training (Fig 18, 20): ~6+ hours")
            print("\nSkip these with --skip-ablations or --skip-multipde")
            if not args.skip_ablations:
                all_experiments.append(expensive_experiments[0])
            if not args.skip_multipde:
                all_experiments.append(expensive_experiments[1])
    
    # Run experiments
    print("\n" + "="*80)
    print(f"RUNNING {len([e for e, *_ in all_experiments if e])} EXPERIMENTS")
    print("="*80)
    
    for should_run, cmd, description in all_experiments:
        if not should_run:
            print(f"\n⊘ Skipped: {description}")
            continue
        
        success = run_command(cmd, description)
        if success:
            completed.append(description)
        else:
            failed.append(description)
    
    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    print(f"\n✅ Completed ({len(completed)}):")
    for desc in completed:
        print(f"  • {desc}")
    
    if failed:
        print(f"\n❌ Failed ({len(failed)}):")
        for desc in failed:
            print(f"  • {desc}")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. Regenerate template figures with real data:
   $ python -m src.analysis.generate_publication_figures \\
       --config config.yaml \\
       --results-dir results \\
       --outdir results/figures

2. Check updated figures:
   $ ls -lh results/figures/

3. Compare with previous versions:
   $ diff results/figures/fig4_rollout_diagnostics.png \\
          results/figures/fig4_rollout_diagnostics.old.png

4. For multi-PDE and ablation figures, custom analysis scripts may be needed
   to integrate data into the figure generation pipeline.
""")
    
    print("="*80 + "\n")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
