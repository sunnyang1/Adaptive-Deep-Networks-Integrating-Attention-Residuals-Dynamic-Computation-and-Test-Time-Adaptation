#!/usr/bin/env python3
"""
Base Experiment Class for all experiments.

Provides common functionality for:
- Configuration management
- Result logging
- Progress tracking
- Error handling
"""

import argparse
import json
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ExperimentConfig:
    """Base configuration for experiments."""

    experiment_id: str = "unknown"
    name: str = "Unknown Experiment"
    description: str = ""
    output_dir: str = "results/experiments"
    seed: int = 42
    device: str = "cpu"
    verbose: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Experiment result container."""

    success: bool
    duration_seconds: float
    metrics: dict[str, Any]
    config: dict[str, Any]
    error_message: str | None = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseExperiment(ABC):
    """
    Base class for all experiments.

    Subclasses should implement:
    - setup(): Initialize experiment resources
    - run(): Execute the experiment
    - teardown(): Clean up resources
    - get_default_config(): Return default configuration
    """

    def __init__(self, config: ExperimentConfig | None = None):
        self.config = config or self.get_default_config()
        self.results: dict[str, Any] = {}
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    @abstractmethod
    def get_default_config(self) -> ExperimentConfig:
        """Return default experiment configuration."""
        pass

    @abstractmethod
    def setup(self) -> None:
        """Set up experiment resources."""
        pass

    @abstractmethod
    def run(self) -> dict[str, Any]:
        """
        Run the experiment.

        Returns:
            Dictionary of experiment results
        """
        pass

    @abstractmethod
    def teardown(self) -> None:
        """Clean up experiment resources. Override if needed."""
        pass

    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message if verbose mode is enabled."""
        if self.config.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")

    def execute(self) -> ExperimentResult:
        """
        Execute the full experiment lifecycle.

        Returns:
            ExperimentResult with success status and metrics
        """
        self.start_time = time.time()
        self.log(f"Starting experiment: {self.config.name}")

        try:
            # Setup phase
            self.log("Setting up experiment...")
            self.setup()

            # Run phase
            self.log("Running experiment...")
            self.results = self.run()

            # Teardown phase
            self.log("Cleaning up...")
            self.teardown()

            self.end_time = time.time()
            duration = self.end_time - self.start_time

            self.log(f"Experiment completed successfully in {duration:.2f}s")

            return ExperimentResult(
                success=True,
                duration_seconds=duration,
                metrics=self.results,
                config=self.config.to_dict(),
            )

        except Exception as e:
            self.end_time = time.time()
            duration = self.end_time - self.start_time

            self.log(f"Experiment failed after {duration:.2f}s: {str(e)}", level="ERROR")

            return ExperimentResult(
                success=False,
                duration_seconds=duration,
                metrics=self.results,
                config=self.config.to_dict(),
                error_message=str(e),
            )

    def save_results(self, result: ExperimentResult, filename: str | None = None) -> Path:
        """
        Save experiment results to JSON file.

        Args:
            result: ExperimentResult to save
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.experiment_id}_{timestamp}.json"

        output_path = output_dir / filename

        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        self.log(f"Results saved to: {output_path}")
        return output_path

    @classmethod
    def create_argument_parser(cls) -> argparse.ArgumentParser:
        """Create argument parser with common arguments."""
        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument(
            "--output-dir",
            type=str,
            default="results/experiments",
            help="Output directory for results",
        )
        parser.add_argument("--seed", type=int, default=42, help="Random seed")
        parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
        parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
        return parser
