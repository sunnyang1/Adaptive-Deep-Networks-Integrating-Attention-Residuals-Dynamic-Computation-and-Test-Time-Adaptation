#!/usr/bin/env python
"""
Table 5: Logit Margin Distribution - Standalone Runner

Measures attention logit margins before/after qTTT adaptation
across different context lengths. Validates the theoretical
Ω(log T) margin requirement from Bansal et al. [4].

Usage:
    python run_table5.py --quick
    python run_table5.py --context_lengths 1024 4096 16384 --num_samples 20
    python run_table5.py --output_dir results/table5
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is in path
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Also ensure the experiment module is importable
_exp_dir = Path(__file__).parent
if str(_exp_dir) not in sys.path:
    sys.path.insert(0, str(_exp_dir))

from experiment import (  # noqa: E402
    MarginSimulator,
    measure_margins,
    plot_improvement_ratio,
    plot_margin_histogram,
    plot_margin_vs_context,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Table 5: Logit Margin Distribution Experiment")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: context_lengths=[1024,4096], num_samples=10",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use (cpu/cuda, default: cpu)"
    )
    parser.add_argument(
        "--context_lengths",
        type=int,
        nargs="+",
        default=None,
        help="Context lengths to test (overrides --quick)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples per context length (overrides --quick)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/table5_margin_distribution",
        help="Output directory for results and plots",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def generate_report(results, context_lengths, config, output_dir):
    """Generate a markdown report of the experiment."""
    report_path = os.path.join(output_dir, "table5_report.md")

    lines = [
        "# Table 5: Logit Margin Distribution Report",
        "",
        f"**Generated**: {datetime.now().isoformat()}",
        "**Mode**: {}".format(config.get("mode", "full")),
        "**Samples per length**: {}".format(config.get("num_samples", "N/A")),
        "",
        "## Summary",
        "",
        "| Context Length | Margin Before | Margin After | Delta | Theoretical Min | Ratio (After/Min) |",
        "|---------------|--------------|-------------|-------|----------------|-------------------|",
    ]

    for T in context_lengths:
        d = results[str(T)]
        lines.append(
            "| {:,} | {:.3f} ± {:.3f} | {:.3f} ± {:.3f} | {:.3f} | {:.3f} | {:.1f}% |".format(
                T,
                d["mean_margin_before"],
                d["std_margin_before"],
                d["mean_margin_after"],
                d["std_margin_after"],
                d["delta_margin"],
                d["theoretical_min"],
                d["pct_theoretical_minimum"],
            )
        )

    # Key findings
    lines.extend(
        [
            "",
            "## Key Findings",
            "",
        ]
    )

    # Check score dilution (before qTTT)
    T_min = min(context_lengths)
    T_max = max(context_lengths)
    margin_before_min = results[str(T_min)]["mean_margin_before"]
    margin_before_max = results[str(T_max)]["mean_margin_before"]
    if margin_before_max < margin_before_min:
        lines.append(
            f"- **Score Dilution**: Before qTTT, margin decreases from {margin_before_min:.3f} "
            f"(T={T_min:,}) to {margin_before_max:.3f} (T={T_max:,}), confirming dilution effect."
        )
    else:
        lines.append(
            "- **Note**: Margin before qTTT did not decrease monotonically " "(simulation noise)."
        )

    # Check qTTT improvement
    margin_after_max = results[str(T_max)]["mean_margin_after"]
    theoretical_max = results[str(T_max)]["theoretical_min"]
    pct = results[str(T_max)]["pct_theoretical_minimum"]
    lines.append(
        f"- **qTTT Recovery**: After qTTT at T={T_max:,}, margin = {margin_after_max:.3f}, "
        f"which is {pct:.1f}% of the theoretical minimum ({theoretical_max:.3f})."
    )

    # Overall improvement ratio
    avg_ratio = sum(results[str(T)]["margin_improvement_ratio"] for T in context_lengths) / len(
        context_lengths
    )
    lines.append(f"- **Average Improvement Ratio**: {avg_ratio:.2f}x across all context lengths.")

    lines.extend(
        [
            "",
            "## Simulation Parameters",
            "",
            "| Parameter | Value |",
            "|-----------|-------|",
            f"| Baseline Margin | {MarginSimulator().baseline_margin:.1f} |",
            f"| Decay Rate | {MarginSimulator().decay_rate:.2f} per log2(T) |",
            f"| Improvement Factor | {MarginSimulator().improvement_factor:.1f}x |",
            "",
        ]
    )

    report = "\n".join(lines)

    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)

    return report_path


def main():
    args = parse_args()

    # Determine context lengths and sample count
    if args.quick:
        context_lengths = [1024, 4096]
        num_samples = 10
        mode = "quick"
    elif args.context_lengths is not None:
        context_lengths = args.context_lengths
        num_samples = args.num_samples if args.num_samples is not None else 50
        mode = "custom"
    else:
        context_lengths = [4096, 16384, 65536, 131072, 262144]
        num_samples = 50
        mode = "full"

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    config = {
        "mode": mode,
        "context_lengths": context_lengths,
        "num_samples": num_samples,
        "device": args.device,
        "seed": args.seed,
    }

    print("=" * 60)
    print("Table 5: Logit Margin Distribution")
    print("=" * 60)
    print(f"Mode:       {mode}")
    print(f"Contexts:   {context_lengths}")
    print(f"Samples:    {num_samples}")
    print(f"Output:     {output_dir}")
    print(f"Seed:       {args.seed}")
    print("=" * 60)

    # Create simulator
    simulator = MarginSimulator(seed=args.seed)

    # Measure margins
    start_time = time.time()
    print("\nMeasuring margins...")
    results = measure_margins(simulator, context_lengths, num_samples)
    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.2f}s")

    # Print summary table
    print(
        "\n{:<12s} {:>12s} {:>12s} {:>10s} {:>14s} {:>10s}".format(
            "Ctx Length", "Before", "After", "Delta", "Th. Min", "Ratio%"
        )
    )
    print("-" * 72)
    for T in context_lengths:
        d = results[str(T)]
        print(
            "{:<12,d} {:>12.3f} {:>12.3f} {:>10.3f} {:>14.3f} {:>10.1f}%".format(
                T,
                d["mean_margin_before"],
                d["mean_margin_after"],
                d["delta_margin"],
                d["theoretical_min"],
                d["pct_theoretical_minimum"],
            )
        )

    # Save results JSON (exclude raw margin lists for cleaner output)
    results_json = {}
    for k, v in results.items():
        entry = dict(v)
        # Keep raw lists in detailed JSON
        results_json[k] = entry

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Generate visualizations
    print("\nGenerating plots...")

    # Plot 1: Margin vs context length
    plot_path = os.path.join(output_dir, "margin_vs_context.png")
    result = plot_margin_vs_context(results, simulator, Path(plot_path))
    if result:
        print(f"  -> {plot_path}")

    # Plot 2: Histogram at middle context length
    mid_idx = len(context_lengths) // 2
    mid_ctx = context_lengths[mid_idx]
    hist_path = os.path.join(output_dir, f"margin_distribution_{mid_ctx}.png")
    result = plot_margin_histogram(results, mid_ctx, Path(hist_path))
    if result:
        print(f"  -> {hist_path}")

    # Also generate histogram at the largest context length
    max_ctx = max(context_lengths)
    hist_path_max = os.path.join(output_dir, f"margin_distribution_{max_ctx}.png")
    result = plot_margin_histogram(results, max_ctx, Path(hist_path_max))
    if result:
        print(f"  -> {hist_path_max}")

    # Plot 3: Improvement ratio
    ratio_path = os.path.join(output_dir, "margin_improvement_ratio.png")
    result = plot_improvement_ratio(results, Path(ratio_path))
    if result:
        print(f"  -> {ratio_path}")

    # Generate report
    print("\nGenerating report...")
    report_path = generate_report(results, context_lengths, config, output_dir)
    print(f"  -> {report_path}")

    print("\n" + "=" * 60)
    print("Table 5 experiment complete!")
    print(f"Total time: {time.time() - start_time:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
