"""
Base Class for Core Experiments

Extends BaseExperiment with simulation capabilities.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from abc import abstractmethod

from experiments.runner import BaseExperiment, ExperimentResult
from experiments.common import ExperimentConfig, OutputPaths, get_device
from experiments.common.visualization import ARCHITECTURE_COLORS, ARCHITECTURE_LABELS


class SimpleTransformer(nn.Module):
    """Simple transformer for experiments."""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        mlp_ratio: int = 4,
        architecture: str = 'prenorm'
    ):
        super().__init__()
        self.architecture = architecture
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * mlp_ratio,
            batch_first=True,
            norm_first=(architecture == 'prenorm')
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.output(x)


class CoreExperiment(BaseExperiment):
    """
    Base class for core paper experiments.
    
    Provides:
    - Model factory for different architectures
    - Standard visualization colors
    - Common measurement utilities
    """
    
    def __init__(self, name: str, config_path: Optional[Path] = None):
        super().__init__(name, category="core")
        self.config_path = config_path
        self.models: Dict[str, nn.Module] = {}
        self.device = None
    
    def setup(self, config: ExperimentConfig) -> None:
        """Setup experiment with configuration."""
        self.config = config
        self.output_paths = OutputPaths.from_config(config)
        self.device = get_device(config.device)
        
        # Load YAML config if provided
        if self.config_path and self.config_path.exists():
            import yaml
            with open(self.config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                self.yaml_config = yaml_config
    
    def create_model(
        self,
        architecture: str,
        num_layers: Optional[int] = None,
        d_model: Optional[int] = None,
        **kwargs
    ) -> nn.Module:
        """
        Factory method to create models.
        
        Args:
            architecture: Architecture type (prenorm, postnorm, deepnorm, attnres)
            num_layers: Number of layers (default from config)
            d_model: Model dimension (default from config)
            **kwargs: Additional arguments
        
        Returns:
            Created model
        """
        if num_layers is None:
            num_layers = getattr(self.config.model_config, 'num_layers', 32)
        if d_model is None:
            d_model = getattr(self.config.model_config, 'hidden_dim', 4096)
        
        vocab_size = getattr(self.config.model_config, 'vocab_size', 32000)
        num_heads = getattr(self.config.model_config, 'num_heads', 32)
        
        model = SimpleTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            architecture=architecture
        )
        
        model = model.to(self.device)
        self.models[architecture] = model
        
        return model
    
    def get_architecture_label(self, arch: str) -> str:
        """Get human-readable label for architecture."""
        return ARCHITECTURE_LABELS.get(arch, arch)
    
    def get_architecture_color(self, arch: str) -> str:
        """Get color for architecture."""
        return ARCHITECTURE_COLORS.get(arch, '#95a5a6')
    
    @abstractmethod
    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """Run the experiment (must be implemented by subclasses)."""
        pass


class ValidationMixin:
    """Mixin for experiments that validate paper targets."""
    
    def validate_target(
        self,
        actual: float,
        target: float,
        tolerance: float = 0.15,
        name: str = ""
    ) -> Dict[str, Any]:
        """
        Validate actual value against target.
        
        Args:
            actual: Actual measured value
            target: Expected target value
            tolerance: Tolerance ratio (default 15%)
            name: Metric name for reporting
        
        Returns:
            Validation result dict
        """
        relative_error = abs(actual - target) / target if target != 0 else abs(actual)
        passed = relative_error <= tolerance
        
        return {
            'name': name,
            'actual': actual,
            'target': target,
            'tolerance': tolerance,
            'relative_error': relative_error,
            'passed': passed,
            'status': 'PASS' if passed else 'FAIL'
        }
    
    def generate_validation_report(
        self,
        validations: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Generate validation report.
        
        Args:
            validations: {metric_name: validation_result}
        
        Returns:
            Markdown-formatted report
        """
        lines = [
            "## Validation Results",
            "",
            "| Metric | Target | Actual | Error | Status |",
            "|--------|--------|--------|-------|--------|",
        ]
        
        all_passed = True
        for name, result in validations.items():
            status_icon = "✅" if result['passed'] else "❌"
            if not result['passed']:
                all_passed = False
            
            lines.append(
                f"| {name} | {result['target']:.3f} | "
                f"{result['actual']:.3f} | "
                f"{result['relative_error']:.1%} | "
                f"{status_icon} |"
            )
        
        lines.extend([
            "",
            f"**Overall**: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}",
        ])
        
        return "\n".join(lines)
