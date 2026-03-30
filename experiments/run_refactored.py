#!/usr/bin/env python3
"""
Refactored Experiment Runner

Demonstrates the new architecture with unified experiment execution.

Usage:
    # Run specific experiment
    python experiments/run_refactored.py exp1_representation_burial
    
    # Run with config file
    python experiments/run_refactored.py exp1 --config configs/experiments/exp1.yaml
    
    # List available experiments
    python experiments/run_refactored.py --list
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from typing import Optional

from experiments.common import ExperimentConfig, get_device, get_logger
from experiments.core.exp1_representation_burial.experiment import \
    RepresentationBurialExperiment
from experiments.core.exp2_margin_analysis.experiment import \
    MarginAnalysisExperiment
from experiments.core.exp3_gradient_flow.experiment import \
    GradientFlowExperiment
from experiments.core.exp4_flop_equivalence.experiment import \
    FLOPEquivalenceExperiment
from experiments.core.exp5_synergy.experiment import \
    ComponentSynergyExperiment
from experiments.core.exp6_auxiliary.experiment import \
    AuxiliaryValidationExperiment


EXPERIMENT_REGISTRY = {
    # Core experiments
    'exp1_representation_burial': RepresentationBurialExperiment,
    'exp1': RepresentationBurialExperiment,
    'exp2_margin_analysis': MarginAnalysisExperiment,
    'exp2': MarginAnalysisExperiment,
    'exp3_gradient_flow': GradientFlowExperiment,
    'exp3': GradientFlowExperiment,
    'exp4_flop_equivalence': FLOPEquivalenceExperiment,
    'exp4': FLOPEquivalenceExperiment,
    'exp5_synergy': ComponentSynergyExperiment,
    'exp5': ComponentSynergyExperiment,
    'exp6_auxiliary': AuxiliaryValidationExperiment,
    'exp6': AuxiliaryValidationExperiment,
}


def list_experiments():
    """List available experiments."""
    print("="*60)
    print("Available Experiments")
    print("="*60)
    
    for name in sorted(set(EXPERIMENT_REGISTRY.keys())):
        exp_class = EXPERIMENT_REGISTRY[name]
        print(f"  - {name}: {exp_class.__doc__.split(chr(10))[0] if exp_class.__doc__ else 'No description'}")


def run_experiment(
    name: str,
    config_path: Optional[Path] = None,
    device: str = 'auto',
    output_dir: Optional[Path] = None,
    quick: bool = False
):
    """Run a single experiment."""
    logger = get_logger('runner')
    
    # Get experiment class
    exp_class = EXPERIMENT_REGISTRY.get(name)
    if exp_class is None:
        logger.error(f"Unknown experiment: {name}")
        logger.info(f"Available: {', '.join(EXPERIMENT_REGISTRY.keys())}")
        return 1
    
    # Create config
    if config_path and config_path.exists():
        config = ExperimentConfig.from_yaml(config_path)
    else:
        config = ExperimentConfig(
            name=name,
            category='core',
            device=device,
            output_dir=output_dir or Path(f'results/core/{name}')
        )
    
    # Quick mode
    if quick:
        config.custom_settings['num_samples'] = 10
        logger.info("Quick mode enabled (reduced samples)")
    
    # Create and run experiment
    experiment = exp_class()
    result = experiment.execute(config)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"Status: {'✅ PASS' if result.success else '❌ FAIL'}")
    print(f"Duration: {result.duration_seconds:.2f}s")
    print(f"Output: {config.output_dir}")
    
    if result.metrics:
        print("\nMetrics:")
        for key, value in result.metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
    
    print('='*60)
    
    return 0 if result.success else 1


def main():
    parser = argparse.ArgumentParser(
        description='Run experiments with refactored architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s exp1_representation_burial
  %(prog)s exp1 --device cuda --quick
  %(prog)s --list
        """
    )
    
    parser.add_argument('experiment', nargs='?',
                       help='Experiment name to run')
    parser.add_argument('--list', action='store_true',
                       help='List available experiments')
    parser.add_argument('--config', type=Path,
                       help='Path to config YAML file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cuda, cpu)')
    parser.add_argument('--output-dir', type=Path,
                       help='Output directory')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (reduced samples)')
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        list_experiments()
        return 0
    
    # Require experiment name
    if not args.experiment:
        parser.print_help()
        return 1
    
    # Run experiment
    return run_experiment(
        name=args.experiment,
        config_path=args.config,
        device=args.device,
        output_dir=args.output_dir,
        quick=args.quick
    )


if __name__ == '__main__':
    exit(main())
