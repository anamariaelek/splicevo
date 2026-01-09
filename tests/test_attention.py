"""Test script to verify attention mechanism implementation."""

import torch
import sys

from splicevo.model.model import MultiHeadAttention, SplicevoModel

def test_attention():
    """Test MultiHeadAttention layer."""
    print("Testing MultiHeadAttention layer...")
    
    # Create attention layer
    attention = MultiHeadAttention(embed_dim=128, num_heads=8, dropout=0.1)
    
    # Create dummy input
    batch_size, seq_len, embed_dim = 16, 1900, 128
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Forward pass
    output = attention(x)
    
    # Check output shape
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    print(f"✓ Attention output shape: {output.shape}")
    
    # Check that gradients flow
    loss = output.sum()
    loss.backward()
    
    for name, param in attention.named_parameters():
        if param.grad is not None:
            print(f"✓ Gradients flow through {name}")
        else:
            print(f"✗ No gradients for {name}")
    
    print("✓ MultiHeadAttention layer test passed!\n")


def test_model_with_attention():
    """Test SplicevoModel with attention layer."""
    print("Testing SplicevoModel with attention...")
    
    # Create model with smaller context_len for testing
    model = SplicevoModel(
        embed_dim=128,
        num_resblocks=4,
        num_classes=3,
        n_conditions=5,
        context_len=450,  # 450 context on each side
        dropout=0.1
    )
    
    # Create dummy input with proper length (1900 = 450 + 1000 + 450)
    batch_size, seq_len = 4, 1900
    sequences = torch.randn(batch_size, seq_len, 4)
    
    # Forward pass
    output = model(sequences, return_features=True)
    
    # Check output keys
    expected_keys = ['splice_logits', 'usage_predictions', 'encoder_features', 'central_features', 'skip_features', 'transformer_features']
    for key in expected_keys:
        assert key in output, f"Missing key: {key}"
        print(f"✓ Output contains '{key}'")
    
    # Check shapes
    print(f"✓ splice_logits shape: {output['splice_logits'].shape}")
    print(f"✓ usage_predictions shape: {output['usage_predictions'].shape}")
    print(f"✓ skip_features shape: {output['skip_features'].shape}")
    
    # Verify skip_features now contains attention output (shape should be full sequence)
    assert output['skip_features'].shape == (batch_size, seq_len, 128), \
        f"Expected skip_features shape {(batch_size, seq_len, 128)}, got {output['skip_features'].shape}"
    
    print("✓ SplicevoModel with attention test passed!\n")


def test_gradient_flow():
    """Test that gradients flow through the entire model."""
    print("Testing gradient flow through model...")
    
    model = SplicevoModel(
        embed_dim=128,
        num_resblocks=4,
        num_classes=3,
        n_conditions=5,
        context_len=450,
        dropout=0.1
    )
    
    # Create dummy data with proper length
    sequences = torch.randn(2, 1900, 4, requires_grad=False)
    
    # Forward pass
    output = model(sequences)
    loss = output['splice_logits'].sum() + output['usage_predictions'].sum()
    
    # Backward pass
    loss.backward()
    
    # Check that transformer attention parameters have gradients
    transformer_params = list(model.transformer.attention.parameters())
    assert len(transformer_params) > 0, "No transformer attention parameters found"
    
    has_gradients = False
    for param in transformer_params:
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break
    
    assert has_gradients, "No gradients flowing through transformer attention layer"
    print("✓ Gradients flow through transformer attention layer")
    print("✓ Gradient flow test passed!\n")


if __name__ == "__main__":
    try:
        test_attention()
        test_model_with_attention()
        test_gradient_flow()
        print("=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
