"""
Tests for TurboQuant V3 (tonbistudio improvements)
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.turboquant import (
    TurboQuantV3,
    TurboQuantV3Config,
    MSECompressor,
    MSECompressorConfig,
    LloydMaxQuantizerV3,
    RandomRotation,
    pack_bits,
    unpack_bits,
    create_v3_k4_v2,
    create_v3_k3_v2,
    create_v3_layer_adaptive,
    V3_RECOMMENDED,
)


class TestBitPacking:
    """Test bit-packing utilities."""
    
    def test_pack_unpack_2bit(self):
        """Test 2-bit packing."""
        tensor = torch.randint(0, 4, (100,))  # 2-bit values
        packed, shape = pack_bits(tensor, bits=2)
        unpacked = unpack_bits(packed, bits=2, original_shape=shape)
        
        assert torch.equal(tensor, unpacked)
    
    def test_pack_unpack_4bit(self):
        """Test 4-bit packing."""
        tensor = torch.randint(0, 16, (100,))  # 4-bit values
        packed, shape = pack_bits(tensor, bits=4)
        unpacked = unpack_bits(packed, bits=4, original_shape=shape)
        
        assert torch.equal(tensor, unpacked)
    
    def test_pack_unpack_3bit(self):
        """Test 3-bit packing."""
        tensor = torch.randint(0, 8, (96,))  # 3-bit values
        packed, shape = pack_bits(tensor, bits=3)
        unpacked = unpack_bits(packed, bits=3, original_shape=shape)
        
        assert torch.equal(tensor, unpacked)


class TestRandomRotation:
    """Test random rotation."""
    
    def test_rotation_isometry(self):
        """Test that rotation preserves norms."""
        rot = RandomRotation(dim=64)
        x = torch.randn(10, 64)
        
        x_rot = rot.rotate(x)
        x_inv = rot.inverse(x_rot)
        
        # Check norm preservation
        torch.testing.assert_close(x.norm(dim=-1), x_rot.norm(dim=-1), rtol=1e-4)
        
        # Check invertibility
        torch.testing.assert_close(x, x_inv, rtol=1e-4, atol=1e-4)
    
    def test_rotation_distribution(self):
        """Test that rotation creates bell-curve distribution."""
        rot = RandomRotation(dim=128)
        
        # Create uniform data
        x = torch.randn(1000, 128)
        x_rot = rot.rotate(x)
        
        # After rotation, each coordinate should be roughly Gaussian
        means = x_rot.mean(dim=0)
        stds = x_rot.std(dim=0)
        
        # Means should be close to 0
        assert means.abs().mean() < 0.1
        
        # Standard deviations should be similar
        assert stds.std() < 0.2


class TestLloydMaxQuantizerV3:
    """Test V3 Lloyd-Max quantizer."""
    
    def test_fit_gaussian(self):
        """Test fitting on Gaussian data."""
        quantizer = LloydMaxQuantizerV3(num_bits=4)
        
        data = torch.randn(10000, 64)
        quantizer.fit(data)
        
        assert quantizer._fitted
        assert len(quantizer.centroids) == 16  # 2^4
    
    def test_quantize_dequantize(self):
        """Test quantization roundtrip."""
        quantizer = LloydMaxQuantizerV3(num_bits=4)
        
        # Fit
        train_data = torch.randn(5000, 32)
        quantizer.fit(train_data)
        
        # Quantize
        test_data = torch.randn(100, 32)
        indices, dequantized = quantizer.quantize(test_data)
        
        assert indices.shape == test_data.shape
        assert dequantized.shape == test_data.shape
        
        # Check error is reasonable
        error = (test_data - dequantized).abs().mean()
        assert error < 0.1
    
    def test_beta_fit(self):
        """Test Beta distribution fitting."""
        quantizer = LloydMaxQuantizerV3(num_bits=3)
        quantizer.fit_beta(alpha=2.0, beta=2.0, scale=1.0)
        
        assert quantizer._fitted
        assert len(quantizer.centroids) == 8


class TestMSECompressor:
    """Test MSE compressor."""
    
    def test_compress_decompress_4bit(self):
        """Test 4-bit compression roundtrip."""
        config = MSECompressorConfig(bits=4, use_rotation=True, pack_bits=True)
        compressor = MSECompressor(config)
        
        # Fit
        sample = torch.randn(100, 64)
        compressor.fit(sample, head_dim=64)
        
        # Compress
        data = torch.randn(10, 64)
        compressed = compressor.compress(data, head_dim=64)
        
        # Decompress
        decompressed = compressor.decompress(compressed)
        
        assert decompressed.shape == data.shape
        
        # Check error
        error = (data - decompressed).abs().mean()
        assert error < 0.1
    
    def test_no_rotation(self):
        """Test compression without rotation."""
        config = MSECompressorConfig(bits=4, use_rotation=False, pack_bits=False)
        compressor = MSECompressor(config)
        
        sample = torch.randn(100, 64)
        compressor.fit(sample, head_dim=64)
        
        data = torch.randn(10, 64)
        compressed = compressor.compress(data, head_dim=64)
        decompressed = compressor.decompress(compressed)
        
        assert decompressed.shape == data.shape


class TestTurboQuantV3:
    """Test V3 main class."""
    
    @pytest.fixture
    def sample_kv(self):
        """Create sample KV cache."""
        return torch.randn(2, 4, 128, 64), torch.randn(2, 4, 128, 64)
    
    def test_k4_v2_compression(self, sample_kv):
        """Test K4/V2 compression."""
        keys, values = sample_kv
        
        v3 = create_v3_k4_v2(head_dim=64)
        v3.fit(keys[:1, :1, :32], values[:1, :1, :32], head_dim=64, layer_idx=0)
        
        compressed = v3.compress_kv(keys, values, head_dim=64, layer_idx=0)
        keys_deq, values_deq = v3.decompress_kv(compressed)
        
        assert keys_deq.shape == keys.shape
        assert values_deq.shape == values.shape
    
    def test_k3_v2_compression(self, sample_kv):
        """Test K3/V2 compression."""
        keys, values = sample_kv
        
        v3 = create_v3_k3_v2(head_dim=64)
        v3.fit(keys[:1, :1, :32], values[:1, :1, :32], head_dim=64, layer_idx=0)
        
        compressed = v3.compress_kv(keys, values, head_dim=64, layer_idx=0)
        keys_deq, values_deq = v3.decompress_kv(compressed)
        
        assert keys_deq.shape == keys.shape
    
    def test_layer_adaptive(self, sample_kv):
        """Test layer-adaptive compression."""
        keys, values = sample_kv
        
        v3 = create_v3_layer_adaptive(
            key_bits=3,
            value_bits=2,
            protected_layers=2,
            total_layers=32
        )
        
        # Protected layer should get more bits
        k_bits_protected, v_bits_protected = v3._get_layer_bits(0)
        k_bits_normal, v_bits_normal = v3._get_layer_bits(5)
        
        assert k_bits_protected >= k_bits_normal
        assert v_bits_protected >= v_bits_normal
    
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        v3 = create_v3_k4_v2()
        
        # K4/V2: (4+2)/2 = 3 bits avg -> 16/3 = 5.33x
        ratio = v3.get_compression_ratio(0)
        assert 5.0 <= ratio <= 5.5
    
    def test_memory_stats(self):
        """Test memory statistics."""
        v3 = create_v3_k4_v2()
        stats = v3.memory_stats(
            seq_len=8192,
            num_layers=1,
            batch_size=1,
            num_heads=32,
            head_dim=128
        )
        
        assert 'original_mb' in stats
        assert 'compressed_mb' in stats
        assert 'compression_ratio' in stats
        assert 'memory_saved_percent' in stats
        
        # K4/V2 should save ~67%
        assert stats['memory_saved_percent'] > 60
    
    def test_multiple_layers(self, sample_kv):
        """Test compression with multiple layers."""
        keys, values = sample_kv
        
        config = TurboQuantV3Config(key_bits=4, value_bits=2, total_layers=4)
        v3 = TurboQuantV3(config)
        
        # Fit multiple layers
        for layer_idx in range(4):
            v3.fit(keys[:1, :1, :32], values[:1, :1, :32], head_dim=64, layer_idx=layer_idx)
        
        # Compress each layer
        for layer_idx in range(4):
            compressed = v3.compress_kv(keys, values, head_dim=64, layer_idx=layer_idx)
            keys_deq, values_deq = v3.decompress_kv(compressed)
            
            assert keys_deq.shape == keys.shape


class TestV3RecommendedConfigs:
    """Test recommended V3 configurations."""
    
    def test_all_configs_valid(self):
        """Test all recommended configs create valid compressors."""
        for name, factory in V3_RECOMMENDED.items():
            v3 = factory(head_dim=64, device='cpu')
            assert isinstance(v3, TurboQuantV3)
    
    def test_k4_v2_quality(self):
        """Test K4/V2 has good reconstruction quality."""
        keys = torch.randn(2, 4, 128, 64)
        values = torch.randn(2, 4, 128, 64)
        
        v3 = create_v3_k4_v2(head_dim=64)
        v3.fit(keys[:1, :1, :32], values[:1, :1, :32], head_dim=64, layer_idx=0)
        
        compressed = v3.compress_kv(keys, values, head_dim=64, layer_idx=0)
        keys_deq, values_deq = v3.decompress_kv(compressed)
        
        key_error = (keys - keys_deq).abs().mean()
        value_error = (values - values_deq).abs().mean()
        
        # K4 should have low error
        assert key_error < 0.05
        assert value_error < 0.1


class TestIntegration:
    """Integration tests."""
    
    def test_v3_vs_quantized(self):
        """Compare V3 with standard quantization."""
        keys = torch.randn(2, 4, 256, 64)
        
        # V3 quantization
        v3 = create_v3_k4_v2(head_dim=64)
        v3.fit(keys[:1, :1, :64], keys[:1, :1, :64], head_dim=64, layer_idx=0)
        compressed = v3.compress_kv(keys, keys, head_dim=64, layer_idx=0)
        keys_v3, _ = v3.decompress_kv(compressed)
        
        # Naive 4-bit quantization
        keys_scaled = keys * 7.5
        keys_int = keys_scaled.round().clamp(-8, 7).to(torch.int8)
        keys_naive = keys_int.float() / 7.5
        
        # V3 should have lower error due to rotation
        error_v3 = (keys - keys_v3).abs().mean()
        error_naive = (keys - keys_naive).abs().mean()
        
        assert error_v3 < error_naive
    
    def test_long_sequence(self):
        """Test with long sequences."""
        seq_lens = [128, 512, 2048]
        
        for seq_len in seq_lens:
            keys = torch.randn(1, 8, seq_len, 64)
            values = torch.randn(1, 8, seq_len, 64)
            
            v3 = create_v3_k4_v2(head_dim=64)
            v3.fit(keys[:, :, :64], values[:, :, :64], head_dim=64, layer_idx=0)
            
            compressed = v3.compress_kv(keys, values, head_dim=64, layer_idx=0)
            keys_deq, values_deq = v3.decompress_kv(compressed)
            
            assert keys_deq.shape == keys.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
