# SoftmaxSumPool: Alternative to Transformer Attention

## Overview

The splicevo model now supports two pooling strategies for sequence modeling:
1. **Multi-head Attention** (default) - The standard transformer attention mechanism
2. **SoftmaxSumPool** - A novel cumulative pooling approach using softmax-weighted cumulative sums

## What is SoftmaxSumPool?

SoftmaxSumPool is a **bidirectional** pooling strategy that computes cumulative weighted sums where weights are derived from softmax exponentials. Each position accumulates information from:
- **All previous positions** (0 to i) - forward accumulation
- **All following positions** (i to sequence length) - backward accumulation

Instead of computing attention between all position pairs (which has O(n²) complexity), it uses cumulative sum operations that are more memory-efficient while still allowing each position to see the entire sequence.

### Key Features

- **Bidirectional**: Each position sees both past and future context
- **Lower parameter count**: ~6% fewer parameters compared to multi-head attention
- **Memory efficient**: Linear complexity instead of quadratic
- **Simpler mechanism**: No learned query/key/value projections
- **Learnable**: Optionally learn the multiplication factor during training

### Mathematical Formulation

For an input sequence `x`, SoftmaxSumPool computes:

```
weights = exp(mult_factor * x)

# Forward: accumulate from positions 0 to i
cum_weights_fwd = cumsum(weights)
output_fwd = cumsum(x * weights) / cum_weights_fwd

# Backward: accumulate from positions i to len
cum_weights_bwd = cumsum(flip(weights))
output_bwd = cumsum(flip(x * weights)) / cum_weights_bwd

# Combine both directions
output = (output_fwd + output_bwd) / 2
```

The `mult_factor` parameter controls the sharpness of the weighting (higher values = sharper).

## Configuration

### Using Attention (Default)

```yaml
model:
  pooling_type: 'attention'
  num_heads: 8
  dropout: 0.4
```

### Using SoftmaxSumPool

```yaml
model:
  pooling_type: 'softmax_pool'
  mult_factor: 5.0              # Controls weighting sharpness
  mult_factor_learnable: false  # Make it learnable if desired
  dropout: 0.4
```

**Note**: SoftmaxSumPool is always bidirectional - each position accumulates from both previous and following positions.

## Parameters

### `pooling_type` (string)
- **Options**: `'attention'` or `'softmax_pool'`
- **Default**: `'attention'`
- **Description**: Choose which pooling strategy to use

### Attention Parameters (when `pooling_type='attention'`)

#### `num_heads` (int)
- **Default**: 8
- **Description**: Number of attention heads
- **Note**: `embed_dim` must be divisible by `num_heads`

### SoftmaxSumPool Parameters (when `pooling_type='softmax_pool'`)

#### `mult_factor` (float)
- **Default**: 1.0
- **Range**: Typically 1.0 to 10.0
- **Description**: Multiplication factor for softmax weighting
  - Lower values (1.0-3.0): Smoother, more distributed weighting
  - Higher values (5.0-10.0): Sharper, more focused weighting

#### `mult_factor_learnable` (bool)
- **Default**: false
- **Description**: Whether `mult_factor` should be learned during training
- **Note**: If true, starts at the specified `mult_factor` value and adjusts during training

## Example Configurations

### Example 1: Fast Training Config with SoftmaxSumPool

```yaml
# configs/training_softmax_pool_fast.yaml
model:
  embed_dim: 64
  num_resblocks: 8
  pooling_type: 'softmax_pool'
  mult_factor: 3.0
  mult_factor_learnable: false
  bottleneck_dim: 64
  dropout: 0.3

training:
  learning_rate: 1.0e-3
  n_epochs: 20
  dataloader:
    batch_size: 64
```

### Example 2: Standard Attention Config

```yaml
# configs/training_transformer.yaml
model:
  embed_dim: 128
  num_resblocks: 16
  pooling_type: 'attention'
  num_heads: 8
  bottleneck_dim: 128
  dropout: 0.4

training:
  learning_rate: 1.0e-4
  n_epochs: 50
  dataloader:
    batch_size: 32
```

### Example 3: Learnable SoftmaxSumPool

```yaml
# configs/training_softmax_learnable.yaml
model:
  embed_dim: 128
  num_resblocks: 16
  pooling_type: 'softmax_pool'
  mult_factor: 5.0
  mult_factor_learnable: true  # Learn optimal sharpness
  bottleneck_dim: 128
  dropout: 0.4

training:
  learning_rate: 1.0e-4
  n_epochs: 50
```

## Training

Use the same training script with your chosen config:

```bash
# With SoftmaxSumPool
python scripts/splicevo_train.py --config configs/training_softmax_pool.yaml

# With Attention (default)
python scripts/splicevo_train.py --config configs/training_transformer.yaml
```

## Performance Comparison

Based on initial tests with a small model (embed_dim=64, 4 residual blocks):

| Pooling Type | Parameters | Memory | Speed |
|--------------|------------|--------|-------|
| Attention    | 262,920    | Higher | Slower |
| SoftmaxPool  | 246,280 (-6.3%) | Lower | Faster |

**Note**: The difference becomes more significant with larger models and longer sequences.

## When to Use Each Strategy

### Use Multi-head Attention when:
- You have sufficient GPU memory
- You want the model to learn complex positional relationships
- Standard architecture is important for your use case
- You need proven performance on similar tasks

### Use SoftmaxSumPool when:
- Memory is limited
- Training speed is a priority
- You're working with very long sequences
- You want a simpler, more interpretable pooling mechanism
- You're experimenting with novel architectures

## Implementation Details

The implementation consists of three main components:

1. **SoftmaxSumPool** - Core pooling layer
2. **PoolingModule** - Wrapper that switches between attention and SoftmaxSumPool
3. **TransformerModule** - Uses PoolingModule internally

All components maintain the same input/output interface, so they're drop-in replacements for each other.

## Testing

Run the test suite to verify the implementation:

```bash
python scripts/test_softmax_pool.py
```

This tests:
- SoftmaxSumPool basic functionality
- PoolingModule switching between modes
- TransformerModule with both pooling types
- Full SplicevoModel with both strategies

## Troubleshooting

### Issue: NaN loss during training
- Try reducing `mult_factor` (start with 1.0 or 2.0)
- Increase dropout
- Check learning rate isn't too high

### Issue: Poor performance compared to attention
- Try increasing `mult_factor` (5.0-8.0)
- Consider using `mult_factor_learnable=true`
- Check if bidirectional context is being utilized effectively

### Issue: Out of memory errors
- SoftmaxSumPool should help, but if still occurring:
  - Reduce `batch_size`
  - Enable `use_amp: true` in config
  - Increase `gradient_accumulation_steps`

## References

The SoftmaxSumPool implementation is inspired by cumulative attention mechanisms and provides an efficient alternative to standard transformer attention for sequential data.

## Code Location

- Implementation: [src/splicevo/model/model.py](../src/splicevo/model/model.py)
- Test script: [scripts/test_softmax_pool.py](../scripts/test_softmax_pool.py)
- Example configs: 
  - [configs/training_transformer.yaml](../configs/training_transformer.yaml)
  - [configs/training_softmax_pool.yaml](../configs/training_softmax_pool.yaml)
