# SplicevoModel Data Flow - Tensor Shapes

This document traces the complete data flow through the SplicevoModel with concrete tensor shapes.

## Example Configuration
Based on `configs/training_full.yaml` (with reduced batch_size for clarity):
- `batch_size = 16` (reduced from 128 for easier visualization)
- `seq_len = 1900` (full sequence including context)
- `context_len = 450` (positions on each end)
- `central_len = 1000` (seq_len - 2*context_len)
- `embed_dim = 128`
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
**fusion_reduce (1x1 conv: 640 → 128)**
```
Shape: (16, 128, 1900)
Memory reduction: 7.8 MB → 1.6 MB (5x compression)
Description: Compress multi-scale information
```

**fusion_activation (ReLU)**
```
Shape: (16, 128, 1900)
Description: Non-linearity
```

**fusion_expand (1x1 conv: 128 → 128)**
```
Shape: (16, 128, 1900)
Description: Final fused multi-scale features
```

### After Transpose
```
Shape: (16, 1900, 128)
Description: Back to sequence-first format for layer norm
Memory: ~1.6 MB
```

### After LayerNorm
```
Shape: (16, 1900, 128)
Description: Normalized across embedding dimension
```

### After Dropout (p=0.5)
```
Shape: (16, 1900, 128)
Description: Regularization applied (50% of activations zeroed during training)
Name: fused_skip
```

---

## 6. Central Region Extraction

### Extract Central Region (remove context)
```
Input:  fused_skip = (16, 1900, 128)
        [:, 450:-450, :]
Output: central_skip = (16, 1000, 128)
Description: Only the middle 1000 positions (predictions region)
Memory: ~0.8 MB (16 * 1000 * 128 * 4 bytes)
Context removed: 450 positions on each side are discarded
```

### Also extract from encoder_features
```
Input:  encoder_features = (16, 1900, 128)
        [:, 450:-450, :]
Output: central_features = (16, 1000, 128)
Description: Alternative features (not used for predictions by default)
```

---

## 7. Output Heads (Applied to Central Region Only)

### Transpose central_skip for Conv1d
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

### Usage Prediction Head

**usage_predictor (1x1 conv: 128 → 15)**
```
Shape: (16, 15, 1000)
Description: 5 conditions × 3 parameters = 15 outputs
Outputs per position:
  - Condition 0: [alpha, beta, sse]
  - Condition 1: [alpha, beta, sse]
  - Condition 2: [alpha, beta, sse]
  - Condition 3: [alpha, beta, sse]
  - Condition 4: [alpha, beta, sse]
```

**After transpose**
```
Shape: (16, 1000, 15)
Description: Flattened usage predictions
```

**After reshape**
```
Shape: (16, 1000, 5, 3)
Description: usage_predictions
- Dim 0: batch (16 sequences)
- Dim 1: position (1000 central positions)
- Dim 2: condition (5 tissues/timepoints)
- Dim 3: parameter [alpha, beta, sse] for Beta distribution
Memory: ~960 KB (16 * 1000 * 5 * 3 * 4 bytes)
```

---

## 8. Final Output Dictionary

```python
output = {
    'splice_logits': (16, 1000, 3),        # Classification logits
    'usage_predictions': (16, 1000, 5, 3), # Beta distribution parameters
    # Optional (if return_features=True):
    'encoder_features': (16, 1900, 128),   # Full sequence features
    'central_features': (16, 1000, 128),   # Central region features
    'skip_features': (16, 1900, 128)       # Fused multi-scale skip (full)
}
```

---

## 9. Prediction Mode Output

### After softmax on splice_logits
```
splice_probabilities: (16, 1000, 3)
Description: Probability distribution over classes
Values: Each position sums to 1.0
Example for position i:
  [0.95, 0.03, 0.02] → 95% none, 3% donor, 2% acceptor
```

### After argmax
```
splice_predictions: (16, 1000)
Description: Predicted class indices (0=none, 1=donor, 2=acceptor)
Values: Integers in {0, 1, 2}
Example: [0, 0, 0, 1, 0, 0, 2, 0, ...]
```

### Final prediction dictionary
```python
predictions = {
    'splice_predictions': (16, 1000),      # Class indices
    'splice_probabilities': (16, 1000, 3), # Class probabilities
    'usage_predictions': (16, 1000, 5, 3)  # [alpha, beta, sse] per condition
}
```

---

## Receptive Field Growth

The receptive field grows with each residual block:

| Block | Dilation | Receptive Field | 
|-------|----------|-----------------|
| Input conv | - | 15 bp |
| ResBlock 0 | 1 | 27 bp |
| ResBlock 1 | 1 | 39 bp | 
| ResBlock 2 | 1 | 51 bp | 
| ResBlock 3 | 1 | 63 bp | 
| ResBlock 4 | 2 | 84 bp | 
| ResBlock 5 | 2 | 105 bp | 
| ResBlock 6 | 2 | 126 bp | 
| ResBlock 7 | 2 | 147 bp | 
| ResBlock 8 | 4 | 186 bp | 
| ResBlock 9 | 4 | 225 bp | 
| ResBlock 10 | 4 | 264 bp | 
| ResBlock 11 | 4 | 303 bp | 
| ResBlock 12 | 8 | 374 bp | 
| ResBlock 13 | 8 | 445 bp | 
| ResBlock 14 | 8 | 516 bp | 
| ResBlock 15 | 8 | **587 bp** |

---