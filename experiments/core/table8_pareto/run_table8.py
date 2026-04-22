#!/usr/bin/env python3
"""
Table 8: Accuracy-Compute Pareto Frontier Runner

Usage:
    python experiments/core/table8_pareto/run_table8.py --quick
    python experiments/core/table8_pareto/run_table8.py --device cpu --output_dir results/table8
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.common import ExperimentConfig  # noqa: E402
from experiments.core.table8_pareto.experiment import ParetoFrontierExperiment  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Table 8: Accuracy-Compute Pareto Frontier")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (20 samples instead of 200)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto/cpu/cuda)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or str(PROJECT_ROOT / "results" / "core" / "table8_pareto")

    config = ExperimentConfig(
        name="table8_pareto",
        category="core",
        device=args.device,
        output_dir=Path(output_dir),
    )

    if args.quick:
        config.custom_settings["quick"] = True

    # Run experiment
    experiment = ParetoFrontierExperiment()
    result = experiment.execute(config)

    if not result.success:
        print("\nExperiment FAILED")
        return 1

    # Build results JSON
    configs = result.metrics["configurations"]
    results_data = {
        "experiment": "table8_pareto",
        "paper_table": 8,
        "quick_mode": result.metrics["quick_mode"],
        "num_samples": result.metrics["num_samples"],
        "pareto_count": result.metrics["pareto_count"],
        "configurations": [
            {
                "id": c["id"],
                "name": c["name"],
                "description": c.get("description", ""),
                "flops": c["flops"],
                "accuracy": c["accuracy"],
                "accuracy_std": c["accuracy_std"],
                "efficiency": c["efficiency"],
                "is_pareto": c["is_pareto"],
            }
            for c in configs
        ],
    }

    # Write results.json
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Print summary table
    print(f"\n{'='*70}")
    print("Table 8: Accuracy-Compute Pareto Frontier — Summary")
    print(f"{'='*70}")
    print(
        f"{'Configuration':<35s} {'FLOPs':>12s} {'Accuracy':>10s} {'Efficiency':>11s} {'Pareto':>7s}"
    )
    print(f"{'-'*35} {'-'*12} {'-'*10} {'-'*11} {'-'*7}")

    for c in configs:
        flops = c["flops"] / 1e12
        acc = c["accuracy"] * 100
        eff = c["efficiency"]
        pareto = "⭐" if c["is_pareto"] else "  "
        print(
            f"{pareto} {c['name']:<33s} {flops:>10.3f}T {acc:>8.2f}% {eff:>10.4f} {'Yes' if c['is_pareto'] else 'No':>7s}"
        )

    print(
        f"\nPareto-optimal: {result.metrics['pareto_count']}/{result.metrics['total_configurations']}"
    )

    # Report
    report = experiment.generate_report(result)
    report_file = output_path / "table8_report.md"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Report saved to: {report_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
