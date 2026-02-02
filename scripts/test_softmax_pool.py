"""
Test script to verify SoftmaxSumPool implementation and integration.
"""

import torch
import sys
sys.path.insert(0, '/home/elek/projects/splicevo/src')

from splicevo.model.model import SoftmaxSumPool, PoolingModule, TransformerModule, SplicevoModel

def test_softmax_sum_pool():
    """Test SoftmaxSumPool basic functionality."""
    print("=" * 60)
    print("Testing SoftmaxSumPool (Bidirectional)")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Test bidirectional pooling
    layer = SoftmaxSumPool(mult_factor=1.0)
    x = torch.randn(2, 10, 8)  # batch_size=2, seq_len=10, embed_dim=8
    y = layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape (bidirectional): {y.shape}")
    assert y.shape == x.shape, "Output shape mismatch!"
    print("✓ Bidirectional pooling works")
    
    # Test learnable parameter
    layer_learnable = SoftmaxSumPool(mult_factor=2.0, mult_factor_learnable=True)
    assert layer_learnable.mult_factor.requires_grad, "mult_factor should be learnable!"
    print("✓ Learnable parameter works")
    
    # Test different mult_factors
    print("\nTesting different mult_factors:")
    for mf in [1.0, 3.0, 5.0]:
        layer = SoftmaxSumPool(mult_factor=mf)
        y = layer(x)
        print(f"  mult_factor={mf}: output range [{y.min():.3f}, {y.max():.3f}]")
    
    print()

def test_pooling_module():
    """Test PoolingModule with both attention and softmax pooling."""
    print("=" * 60)
    print("Testing PoolingModule")
    print("=" * 60)
    
    embed_dim = 128
    batch_size = 2
    seq_len = 100
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Test attention mode
    print("\n--- Testing Attention Mode ---")
    pooling_attention = PoolingModule(
        embed_dim=embed_dim,
        pooling_type='attention',
        num_heads=8,
        dropout=0.1
    )
    y_attention = pooling_attention(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape (attention): {y_attention.shape}")
    assert y_attention.shape == x.shape, "Output shape mismatch!"
    print("✓ Attention pooling works")
    
    # Test softmax pooling mode
    print("\n--- Testing SoftmaxPool Mode (Bidirectional) ---")
    pooling_softmax = PoolingModule(
        embed_dim=embed_dim,
        pooling_type='softmax_pool',
        mult_factor=5.0,
        dropout=0.1
    )
    y_softmax = pooling_softmax(x)
    print(f"Output shape (softmax_pool): {y_softmax.shape}")
    assert y_softmax.shape == x.shape, "Output shape mismatch!"
    print("✓ SoftmaxPool works")
    
    print()

def test_transformer_module():
    """Test TransformerModule with both pooling types."""
    print("=" * 60)
    print("Testing TransformerModule")
    print("=" * 60)
    
    embed_dim = 128
    batch_size = 2
    seq_len = 100
    
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Test with attention
    print("\n--- Testing with Attention ---")
    transformer_attn = TransformerModule(
        embed_dim=embed_dim,
        pooling_type='attention',
        num_heads=8,
        dropout=0.1
    )
    y_attn = transformer_attn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape (attention): {y_attn.shape}")
    assert y_attn.shape == x.shape, "Output shape mismatch!"
    print("✓ TransformerModule with attention works")
    
    # Test with softmax pooling
    print("\n--- Testing with SoftmaxPool (Bidirectional) ---")
    transformer_smp = TransformerModule(
        embed_dim=embed_dim,
        pooling_type='softmax_pool',
        mult_factor=3.0,
        dropout=0.1
    )
    y_smp = transformer_smp(x)
    print(f"Output shape (softmax_pool): {y_smp.shape}")
    assert y_smp.shape == x.shape, "Output shape mismatch!"
    print("✓ TransformerModule with softmax_pool works")
    
    print()

def test_splicevo_model():
    """Test full SplicevoModel with both pooling types."""
    print("=" * 60)
    print("Testing Full SplicevoModel")
    print("=" * 60)
    
    batch_size = 2
    seq_len = 1000  # 100 central + 450*2 context
    
    # One-hot encoded sequences (ACGT = 4 channels)
    sequences = torch.randn(batch_size, seq_len, 4)
    
    # Test with attention
    print("\n--- Testing Model with Attention ---")
    model_attn = SplicevoModel(
        embed_dim=64,
        num_resblocks=4,
        dilation_strategy='exponential',
        num_classes=3,
        n_conditions=5,
        context_len=450,
        pooling_type='attention',
        num_heads=4,
        dropout=0.1,
        bottleneck_dim=64
    )
    
    output_attn = model_attn(sequences)
    print(f"Input shape: {sequences.shape}")
    print(f"Splice logits shape: {output_attn['splice_logits'].shape}")
    print(f"Usage predictions shape: {output_attn['usage_predictions'].shape}")
    
    expected_central_len = seq_len - 2 * 450
    assert output_attn['splice_logits'].shape == (batch_size, expected_central_len, 3)
    assert output_attn['usage_predictions'].shape == (batch_size, expected_central_len, 5)
    print("✓ Model with attention produces correct output shapes")
    
    # Test with softmax pooling
    print("\n--- Testing Model with SoftmaxPool (Bidirectional) ---")
    model_smp = SplicevoModel(
        embed_dim=64,
        num_resblocks=4,
        dilation_strategy='exponential',
        num_classes=3,
        n_conditions=5,
        context_len=450,
        pooling_type='softmax_pool',
        mult_factor=5.0,
        mult_factor_learnable=False,
        dropout=0.1,
        bottleneck_dim=64
    )
    
    output_smp = model_smp(sequences)
    print(f"Splice logits shape: {output_smp['splice_logits'].shape}")
    print(f"Usage predictions shape: {output_smp['usage_predictions'].shape}")
    
    assert output_smp['splice_logits'].shape == (batch_size, expected_central_len, 3)
    assert output_smp['usage_predictions'].shape == (batch_size, expected_central_len, 5)
    print("✓ Model with softmax_pool produces correct output shapes")
    
    # Compare parameter counts
    n_params_attn = sum(p.numel() for p in model_attn.parameters())
    n_params_smp = sum(p.numel() for p in model_smp.parameters())
    
    print(f"\nParameter comparison:")
    print(f"  Attention model: {n_params_attn:,} parameters")
    print(f"  SoftmaxPool model: {n_params_smp:,} parameters")
    print(f"  Difference: {abs(n_params_attn - n_params_smp):,} parameters")
    print(f"  SoftmaxPool is {100 * (1 - n_params_smp/n_params_attn):.1f}% smaller" if n_params_smp < n_params_attn else "")
    
    print()

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SoftmaxSumPool Implementation Tests")
    print("=" * 60 + "\n")
    
    try:
        test_softmax_sum_pool()
        test_pooling_module()
        test_transformer_module()
        test_splicevo_model()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED!")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
