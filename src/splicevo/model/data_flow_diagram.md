# SplicevoModel Data Flow - Tensor Shapes

This document traces the complete data flow through the SplicevoModel with concrete tensor shapes.

## Example Configuration
Based on `configs/training_full.yaml` (with reduced batch_size for clarity):
- `batch_size = 16` (reduced from 128 for easier visualization)
- `seq_len = 1900` (full sequence including context)
- `context_len = 450` (positions on each end)
- `central_len = 1000` (seq_len - 2*context_len)
- `embed_dim = 128`
- `num_heads = 8`
- `num_classes = 3` (none, donor, acceptor)
- `n_conditions = 5` (tissue/timepoint conditions)
- `num_resblocks = 16` 
- `dilation_strategy = alternating` and `alternate = 4` results in dilations: `[1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8]`
- `kernel_sizes = 9` (default)

**Note:** Training uses batch_size=64, but we use 16 here for clearer documentation.

---

## 1. Input Stage

### Input Sequence
```
Shape: (16, 1900, 4)
Description: One-hot encoded DNA sequences
Format: [ACGT] encoding
Memory: ~0.6 MB (16 * 1900 * 4 * 4 bytes)
```

### After Transpose for Conv1d
```
Shape: (16, 4, 1900)
Description: Channels-first format for PyTorch Conv1d
```

---

## 2. Initial Convolution & Embedding

### After input_conv (Conv1d: 4 → 128, k=15, p=7)
```
Shape: (16, 128, 1900)
Description: Projects from 4 nucleotides to 128-dimensional embedding
Memory: ~15 MB (16 * 128 * 1900 * 4 bytes)
Receptive field: 15 positions
```

### After input_relu
```
Shape: (16, 128, 1900)
Description: Non-linearity applied
```

---

## 3. Multi-Scale Skip Connection Collection

### Initial Skip (via initial_skip_proj: 1x1 conv)
```
Shape: (16, 128, 1900)
Description: First skip connection from embedding
Scale: Base embedding (finest resolution)
```

---

## 4. Residual Block Processing

### ResBlock Architecture (per block):
```
Input:  (16, 128, 1900)
  ↓
BatchNorm → ReLU → Conv1d (k=9, dilation=d)
  ↓  
  (16, 128, 1900)
  ↓
BatchNorm → ReLU → Conv1d (k=3, dilation=1)
  ↓
  (16, 128, 1900)
  ↓
Add residual
  ↓
Output: (16, 128, 1900)
```

### Block-by-Block Flow (Alternating Dilation Strategy, alternate=4):

**ResBlock 0-3: dilation=1 (kernel=9)**
- Input: (16, 128, 1900)
- Output: (16, 128, 1900)
- Receptive field growth:
    - Block 0: 15 + (9-1)*1 + (3-1)*1 = 25 positions
    - Block 1: 25 + (9-1)*1 + (3-1)*1 = 35 positions
    - Block 2: 35 + (9-1)*1 + (3-1)*1 = 45 positions
    - Block 3: 45 + (9-1)*1 + (3-1)*1 = 55 positions
- **Scale Group 0 ends here** (after 4 blocks with d=1)
- Skip collected via group_skip_projections[0]: (16, 128, 1900)

**ResBlock 4-7: dilation=2 (kernel=9)**
- Input: (16, 128, 1900)
- Output: (16, 128, 1900)
- Receptive field growth:
    - Block 4: 55 + (9-1)*2 + (3-1)*2 = 75 positions
    - Block 5: 75 + (9-1)*2 + (3-1)*2 = 95 positions
    - Block 6: 95 + (9-1)*2 + (3-1)*2 = 115 positions
    - Block 7: 115 + (9-1)*2 + (3-1)*2 = 135 positions
- **Scale Group 1 ends here** (after 4 blocks with d=2)
- Skip collected via group_skip_projections[1]: (16, 128, 1900)

**ResBlock 8-11: dilation=4 (kernel=9)**
- Input: (16, 128, 1900)
- Output: (16, 128, 1900)
- Receptive field growth:
    - Block 8: 135 + (9-1)*4 + (3-1)*4 = 175 positions
    - Block 9: 175 + (9-1)*4 + (3-1)*4 = 215 positions
    - Block 10: 215 + (9-1)*4 + (3-1)*4 = 255 positions
    - Block 11: 255 + (9-1)*4 + (3-1)*4 = 295 positions
- **Scale Group 2 ends here** (after 4 blocks with d=4)
- Skip collected via group_skip_projections[2]: (16, 128, 1900)

**ResBlock 12-15: dilation=8 (kernel=9)**
- Input: (16, 128, 1900)
- Output: (16, 128, 1900)
- Receptive field growth:
    - Block 12: 295 + (9-1)*8 + (3-1)*8 = 375 positions
    - Block 13: 375 + (9-1)*8 + (3-1)*8 = 455 positions
    - Block 14: 455 + (9-1)*8 + (3-1)*8 = 535 positions
    - Block 15: 535 + (9-1)*8 + (3-1)*8 = 615 positions
- **Scale Group 3 ends here** (after 4 blocks with d=8)
- Skip collected via group_skip_projections[3]: (16, 128, 1900)

---

## 5. Multi-Scale Feature Fusion

### Skip Features Collection
```
skip_features = [
    initial_skip:              (16, 128, 1900)  # Scale 0: embedding
    group_skip_projections[0]: (16, 128, 1900)  # Scale 1: d=1,1,1,1 (blocks 0-3)
    group_skip_projections[1]: (16, 128, 1900)  # Scale 2: d=2,2,2,2 (blocks 4-7)
    group_skip_projections[2]: (16, 128, 1900)  # Scale 3: d=4,4,4,4 (blocks 8-11)
    group_skip_projections[3]: (16, 128, 1900)  # Scale 4: d=8,8,8,8 (blocks 12-15)
]
Total scales: 5
```

### After Concatenation
```
5 scales * 128 channels = 640 channels
Shape: (16, 640, 1900)
Memory: ~7.8 MB (16 * 640 * 1900 * 4 bytes)
Description: All multi-scale features stacked together
```

### After Bottleneck Fusion

**fusion_reduce (1x1 conv: 640 → bottleneck_dim)**
```
Shape: (16, bottleneck_dim, 1900)
Default bottleneck_dim: 128
Alternatives: 256 (balanced), 320 (half), 640 (no compression)
Memory reduction: 7.8 MB → 1.6 MB (5x compression at 128 dims)
Description: Compress multi-scale information into bottleneck
```

**fusion_activation (ReLU)**
```
Shape: (16, bottleneck_dim, 1900)
Description: Non-linear transformation in compressed space
```

**fusion_expand (1x1 conv: bottleneck_dim → bottleneck_dim)**
```
Shape: (16, bottleneck_dim, 1900)
Description: Learn flexible transformation of compressed features
           This second convolution + ReLU creates a two-layer MLP
           that can learn non-linear combinations of multi-scale features
```

### After Transpose
```
Shape: (16, 1900, bottleneck_dim)
Description: Back to sequence-first format for layer norm
Memory: ~1.6 MB (for bottleneck_dim=128)
```

### After LayerNorm
```
Shape: (16, 1900, bottleneck_dim)
Description: Normalized across embedding dimension
```

### After Dropout (p=0.5)
```
Shape: (16, 1900, bottleneck_dim)
Description: Regularization applied (50% of activations zeroed during training)
Name: fused_skip (full sequence, ready for transformer)
```

---

## 6. Multi-Head Self-Attention Layer (Transformer on Full Sequence)

### Input to Attention
```
Shape: (16, 1900, 128)
Description: Full sequence features including context (sequence-first format)
Name: full_features (from encoder output)
Memory: ~1.6 MB (16 * 1900 * 128 * 4 bytes)
```

### Multi-Head Self-Attention
```
Query, Key, Value projections:
- Input: (16, 1900, 128)
- Output: (16, 1900, 128) per projection
- num_heads = 8
- head_dim = 128 / 8 = 16

Attention computation per head:
- Q: (16, 1900, 16) after split into 8 heads
- K: (16, 1900, 16) after split into 8 heads
- V: (16, 1900, 16) after split into 8 heads
- Attention scores: (16, 1900, 1900) per head [seq_len x seq_len]
  * Full sequence attention captures long-range dependencies across entire input
  * Includes context regions in attention mechanism
- Attention weights: softmax applied across sequence dimension
- Attention output per head: (16, 1900, 16)
Memory for attention: ~183 MB for (16, 1900, 1900) float32
Total for 8 heads: ~183 MB (shared computation)
```

### After Head Concatenation
```
Shape: (16, 1900, 128)
Description: Concatenated outputs from 8 attention heads
Memory: ~1.6 MB (16 * 1900 * 128 * 4 bytes)
```

### After Output Projection (1x1 linear)
```
Shape: (16, 1900, 128)
Description: Final attention output
```

### After Residual Connection (add input)
```
Shape: (16, 1900, 128)
Description: full_features + attention_output
Preserves original features while integrating attention-weighted information
Memory: ~1.6 MB
```

### After Layer Normalization
```
Shape: (16, 1900, 128)
Description: Normalized attention output across full sequence
Name: transformer_output
```

### After Dropout (p=0.5)
```
Shape: (16, 1900, 128)
Description: Regularization applied (50% of activations zeroed during training)
Name: transformer_output (full sequence)
```

---

## 7. Central Region Extraction (After Transformer)

### Extract Central Region from Transformer Output
```
Input:  transformer_output = (16, 1900, 128)
        [:, 450:-450, :]
Output: central_features = (16, 1000, 128)
Description: Extract only the middle 1000 positions (prediction region)
Memory: ~0.8 MB (16 * 1000 * 128 * 4 bytes)
Why now: Transformer has already incorporated context information via attention
         across the full sequence. Now extract only central region for predictions.
Context used: 450 positions on each side were included in attention computation,
              allowing central positions to attend to full sequence context.
```

---

## 8. Output Heads (Applied to Central Region)

### Transpose central_features for Conv1d
```
Shape: (16, 128, 1000)
Description: Channels-first for output convolutions
```

### Splice Site Classification Head

**splice_classifier (1x1 conv: 128 → 3)**
```
Shape: (16, 3, 1000)
Description: Logits for 3 classes per position
- Channel 0: "none" (not a splice site)
- Channel 1: "donor" (GT splice donor)
- Channel 2: "acceptor" (AG splice acceptor)
```

**After transpose**
```
Shape: (16, 1000, 3)
Description: splice_logits
Memory: ~192 KB (16 * 1000 * 3 * 4 bytes)
```

### Usage Prediction Head (Regression)

**usage_predictor (1x1 conv: 128 → 5)**
```
Shape: (16, 5, 1000)
Description: SSE for 5 conditions
Outputs per position: [cond0_sse, cond1_sse, cond2_sse, cond3_sse, cond4_sse]
```

**After transpose**
```
Shape: (16, 1000, 5)
Description: SSE predictions
```

**After sigmoid**
```
Shape: (16, 1000, 5)
Description: usage_predictions (SSE in [0,1] range)
- Dim 0: batch (16 sequences)
- Dim 1: position (1000 central positions)
- Dim 2: condition (5 tissues/timepoints)
Memory: ~320 KB (16 * 1000 * 5 * 4 bytes)
```

### Usage Classification Head (for Hybrid Loss)

**usage_classifier (1x1 conv: 128 → 15)** [only if usage_loss_type='hybrid']
```
Shape: (16, 15, 1000)
Description: 5 conditions × 3 classes = 15 outputs
For each condition, predict 3 classes for SSE:
  - is_zero (target < 0.05)
  - is_one (target > 0.95)
  - is_middle (0.05 <= target <= 0.95)
```

**After transpose and reshape**
```
Shape: (16, 1000, 5, 3)
Description: usage_class_logits
- Dim 0: batch (16 sequences)
- Dim 1: position (1000 central positions)
- Dim 2: condition (5 tissues/timepoints)
- Dim 3: class [is_zero, is_one, is_middle]
Memory: ~960 KB (16 * 1000 * 5 * 3 * 4 bytes)
```

---

## 9. Final Output Dictionary

```python
# Without hybrid loss:
output = {
    'splice_logits': (16, 1000, 3),        # Classification logits
    'usage_predictions': (16, 1000, 5),    # Regression: SSE in [0,1]
    # Optional (if return_features=True):
    'encoder_features': (16, 1900, 128),   # Full sequence features from encoder
    'central_features': (16, 1000, 128),   # Central region from transformer output
    'transformer_features': (16, 1900, 128), # Full transformer output
    'skip_features': (16, 1900, 128)       # Fused multi-scale skip (full)
}

# With hybrid loss (usage_loss_type='hybrid'):
output = {
    'splice_logits': (16, 1000, 3),             # Classification logits
    'usage_predictions': (16, 1000, 5),         # Regression: SSE in [0,1]
    'usage_class_logits': (16, 1000, 5, 3),     # Classification: [is_zero, is_one, is_middle]
    # Optional (if return_features=True):
    'encoder_features': (16, 1900, 128),        # Full sequence features from encoder
    'central_features': (16, 1000, 128),        # Central region from transformer output
    'transformer_features': (16, 1900, 128),    # Full transformer output
    'skip_features': (16, 1900, 128)            # Fused multi-scale skip (full)
}
```

---