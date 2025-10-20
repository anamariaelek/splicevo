# SplicevoModel Data Flow - Tensor Shapes

This document traces the complete data flow through the SplicevoModel with concrete tensor shapes.

## Example Configuration
- `batch_size = 8`
- `seq_len = 10000` (full sequence including context)
- `context_len = 4500` (positions on each end)
- `central_len = 1000` (seq_len - 2*context_len)
- `embed_dim = 256`
- `num_classes = 3` (none, donor, acceptor)
- `n_conditions = 5` (tissue/timepoint conditions)
- `num_resblocks = 4` with exponential dilation [1, 2, 4, 8]
- `kernel_sizes = [9, 9, 9, 9]`

---

## 1. Input Stage

### Input Sequence
```
Shape: (8, 10000, 4)
Description: One-hot encoded DNA sequences
Format: [ACGT] encoding
Memory: ~1.3 MB (8 * 10000 * 4 * 4 bytes)
```

### After Transpose for Conv1d
```
Shape: (8, 4, 10000)
Description: Channels-first format for PyTorch Conv1d
```

---

## 2. Initial Convolution & Embedding

### After input_conv (Conv1d: 4 → 256, k=15, p=7)
```
Shape: (8, 256, 10000)
Description: Projects from 4 nucleotides to 256-dimensional embedding
Memory: ~82 MB (8 * 256 * 10000 * 4 bytes)
Receptive field: 15 positions
```

### After input_relu
```
Shape: (8, 256, 10000)
Description: Non-linearity applied
```

---

## 3. Multi-Scale Skip Connection Collection

### Initial Skip (via initial_skip_proj: 1x1 conv)
```
Shape: (8, 256, 10000)
Description: First skip connection from embedding
Scale: Base embedding (finest resolution)
```

---

## 4. Residual Block Processing

### ResBlock Architecture (per block):
```
Input:  (8, 256, 10000)
  ↓
BatchNorm → ReLU → Conv1d (k=9, dilation=d)
  ↓  
  (8, 256, 10000)
  ↓
BatchNorm → ReLU → Conv1d (k=3, dilation=1)
  ↓
  (8, 256, 10000)
  ↓
Add residual
  ↓
Output: (8, 256, 10000)
```

### Block-by-Block Flow:

**ResBlock 0: dilation=1 (kernel=9)**
- Input: (8, 256, 10000)
- Output: (8, 256, 10000)
- Receptive field: 15 + (9*1) + 3 = 27 positions
- **Scale Group 0 ends here**
- Skip collected via group_skip_projections[0]: (8, 256, 10000)

**ResBlock 1: dilation=2 (kernel=9)**
- Input: (8, 256, 10000)
- Output: (8, 256, 10000)
- Receptive field: 27 + (9*2) + 3 = 48 positions
- **Scale Group 1 ends here**
- Skip collected via group_skip_projections[1]: (8, 256, 10000)

**ResBlock 2: dilation=4 (kernel=9)**
- Input: (8, 256, 10000)
- Output: (8, 256, 10000)
- Receptive field: 48 + (9*4) + 3 = 87 positions
- **Scale Group 2 ends here**
- Skip collected via group_skip_projections[2]: (8, 256, 10000)

**ResBlock 3: dilation=8 (kernel=9)**
- Input: (8, 256, 10000)
- Output: (8, 256, 10000)
- Receptive field: 87 + (9*8) + 3 = 162 positions
- **Scale Group 3 ends here**
- Skip collected via group_skip_projections[3]: (8, 256, 10000)

---

## 5. Multi-Scale Feature Fusion

### Skip Features Collection
```
skip_features = [
    initial_skip:              (8, 256, 10000)  # Scale 0: embedding
    group_skip_projections[0]: (8, 256, 10000)  # Scale 1: dilation=1
    group_skip_projections[1]: (8, 256, 10000)  # Scale 2: dilation=2
    group_skip_projections[2]: (8, 256, 10000)  # Scale 3: dilation=4
    group_skip_projections[3]: (8, 256, 10000)  # Scale 4: dilation=8
]
Total scales: 5
```

### After Concatenation
```
5 scales * 256 channels = 1280 channels
Shape: (8, 1280, 10000)
Memory: ~410 MB (8 * 1280 * 10000 * 4 bytes)
```

### After Bottleneck Fusion
**fusion_reduce (1x1 conv: 1280 → 256)**
```
Shape: (8, 256, 10000)
Memory reduction: 410 MB → 82 MB (5x compression)
```

**fusion_activation (ReLU)**
```
Shape: (8, 256, 10000)
```

**fusion_expand (1x1 conv: 256 → 256)**
```
Shape: (8, 256, 10000)
Description: Final fused multi-scale features
```

### After Transpose
```
Shape: (8, 10000, 256)
Description: Back to sequence-first format
Memory: ~82 MB
```

### After LayerNorm
```
Shape: (8, 10000, 256)
Description: Normalized across embedding dimension
```

### After Dropout
```
Shape: (8, 10000, 256)
Description: Regularization applied
Name: fused_skip
```

---

## 6. Central Region Extraction

### Extract Central Region (remove context)
```
Input:  fused_skip = (8, 10000, 256)
        [:, 4500:-4500, :]
Output: central_skip = (8, 1000, 256)
Description: Only the middle 1000 positions (predictions region)
Memory: ~8.2 MB (8 * 1000 * 256 * 4 bytes)
```

### Also extract from encoder_features
```
Input:  encoder_features = (8, 10000, 256)
        [:, 4500:-4500, :]
Output: central_features = (8, 1000, 256)
Description: Alternative features (not used for predictions by default)
```

---

## 7. Output Heads (Applied to Central Region Only)

### Transpose central_skip for Conv1d
```
Shape: (8, 256, 1000)
Description: Channels-first for output convolutions
```

### Splice Site Classification Head

**splice_classifier (1x1 conv: 256 → 3)**
```
Shape: (8, 3, 1000)
Description: Logits for 3 classes (none, donor, acceptor)
```

**After transpose**
```
Shape: (8, 1000, 3)
Description: splice_logits
Memory: ~96 KB (8 * 1000 * 3 * 4 bytes)
```

### Usage Prediction Head

**usage_predictor (1x1 conv: 256 → 15)**
```
Shape: (8, 15, 1000)
Description: 5 conditions × 3 parameters = 15 outputs
```

**After transpose**
```
Shape: (8, 1000, 15)
Description: Flattened usage predictions
```

**After reshape**
```
Shape: (8, 1000, 5, 3)
Description: usage_predictions
- Dim 2: 5 conditions (tissues/timepoints)
- Dim 3: 3 parameters [alpha, beta, sse]
Memory: ~480 KB (8 * 1000 * 5 * 3 * 4 bytes)
```

---

## 8. Final Output Dictionary

```python
output = {
    'splice_logits': (8, 1000, 3),        # Classification logits
    'usage_predictions': (8, 1000, 5, 3), # Beta distribution parameters
    
    # Optional (if return_features=True):
    'encoder_features': (8, 10000, 256),  # Full sequence features
    'central_features': (8, 1000, 256),   # Central region features
    'skip_features': (8, 10000, 256)      # Fused multi-scale skip (full)
}
```

---

## 9. Prediction Mode Output (via .predict())

### After softmax on splice_logits
```
splice_probabilities: (8, 1000, 3)
Description: Probability distribution over classes
Values: Each position sums to 1.0
```

### After argmax
```
splice_predictions: (8, 1000)
Description: Predicted class indices (0=none, 1=donor, 2=acceptor)
Values: Integers in {0, 1, 2}
```

### Final prediction dictionary
```python
predictions = {
    'splice_predictions': (8, 1000),      # Class indices
    'splice_probabilities': (8, 1000, 3), # Class probabilities
    'usage_predictions': (8, 1000, 5, 3)  # [alpha, beta, sse] per condition
}
```

---

## Memory Summary

**Peak memory usage (during forward pass):**

1. Input: ~1.3 MB
2. Embedding: ~82 MB
3. ResBlock processing: ~82 MB (reused)
4. Skip concatenation: ~410 MB (peak!)
5. After bottleneck: ~82 MB
6. Central region: ~8.2 MB
7. Outputs: ~0.6 MB

**Total peak: ~410 MB** (during skip concatenation)
**After fusion: ~82 MB** (5x compression achieved)

---

## Receptive Field Growth

- Initial embedding: 15 positions
- After ResBlock 0 (d=1): 27 positions
- After ResBlock 1 (d=2): 48 positions
- After ResBlock 2 (d=4): 87 positions
- After ResBlock 3 (d=8): 162 positions

**Final receptive field: 162 nucleotides**

This means each output position can "see" ±81 positions around it, capturing local splice site motifs and nearby regulatory elements.

---

## Key Design Decisions

1. **Multi-scale skip connections**: Capture features at different dilation scales
2. **Bottleneck fusion**: Compress 5 scales from 1280 → 256 channels (5x memory reduction)
3. **Context removal**: Only predict on central region, use flanking sequence as context
4. **Separate heads**: Independent 1x1 convolutions for classification and regression
5. **Pre-activation ResBlocks**: Better gradient flow and feature reuse
