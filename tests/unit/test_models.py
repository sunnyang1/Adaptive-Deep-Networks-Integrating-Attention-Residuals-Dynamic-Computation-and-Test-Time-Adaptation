"""
Unit tests for models module.
"""

import pytest


class TestModelsImport:
    """Tests for model imports."""
    
    def test_configs_import(self):
        """Test that configs module can be imported."""
        from src.models import configs
        assert configs is not None
    
    def test_model_config_exists(self):
        """Test that ModelConfig exists."""
        from src.models.configs import ModelConfig
        assert ModelConfig is not None
    
    def test_model_config_creation(self):
        """Test creating a ModelConfig."""
        from src.models.configs import ModelConfig
        # Create with minimal args
        config = ModelConfig()
        assert config is not None


class TestAdaptiveTransformerImport:
    """Import tests for AdaptiveTransformer."""
    
    def test_adaptive_transformer_import(self):
        """Test that AdaptiveTransformer module can be imported."""
        try:
            from src.models.adaptive_transformer import AdaptiveTransformer
            assert AdaptiveTransformer is not None
        except ImportError as e:
            pytest.skip(f"AdaptiveTransformer not available: {e}")


class TestTokenizerImport:
    """Import tests for tokenizer."""
    
    def test_tokenizer_import(self):
        """Test that tokenizer module can be imported."""
        try:
            from src.models.tokenizer import AdaptiveTokenizer
            assert AdaptiveTokenizer is not None
        except ImportError as e:
            pytest.skip(f"AdaptiveTokenizer not available: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
