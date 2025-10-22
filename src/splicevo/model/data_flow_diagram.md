# SplicevoModel Data Flow - Tensor Shapes

This document traces the complete data flow through the SplicevoModel with concrete tensor shapes.

## Example Configuration
Based on `configs/training_default.yaml` (with reduced batch_size for clarity):
- `batch_size = 16` (reduced from 128 for easier visualization)
- `seq_len = 10000` (full sequence including context)
- `context_len = 4500` (positions on each end)
- `central_len = 1000` (seq_len - 2*context_len)
- `embed_dim = 128`
- `num_classes = 3` (none, donor, acceptor)
- `n_conditions = 5` (tissue/timepoint conditions)
- `num_resblocks = 8` with alternating dilation strategy (alternate=2)
- `dilation_strategy = alternating` and `alternate = 2` results in dilations: `[1, 1, 2, 2, 4, 4, 8, 8]`
- `kernel_sizes = [9, 9, 9, 9, 9, 9, 9, 9]` (default)

**Note:** Training uses batch_size=128, but we use 16 here for clearer documentation.

---

## 1. Input Stage

### Input Sequence
```
Shape: (16, 10000, 4)
Description: One-hot encoded DNA sequences
Format: [ACGT] encoding
Memory: ~2.6 MB (16 * 10000 * 4 * 4 bytes)
```

### After Transpose for Conv1d
```
Shape: (16, 4, 10000)
Description: Channels-first format for PyTorch Conv1d
```

---

## 2. Initial Convolution & Embedding

### After input_conv (Conv1d: 4 → 128, k=15, p=7)
```
Shape: (16, 128, 10000)
Description: Projects from 4 nucleotides to 128-dimensional embedding
Memory: ~82 MB (16 * 128 * 10000 * 4 bytes)
Receptive field: 15 positions
```

### After input_relu
```
Shape: (16, 128, 10000)
Description: Non-linearity applied
```

---

## 3. Multi-Scale Skip Connection Collection

### Initial Skip (via initial_skip_proj: 1x1 conv)
```
Shape: (16, 128, 10000)
Description: First skip connection from embedding
Scale: Base embedding (finest resolution)
```

---

## 4. Residual Block Processing

### ResBlock Architecture (per block):
```
Input:  (16, 128, 10000)
  ↓
BatchNorm → ReLU → Conv1d (k=9, dilation=d)
  ↓  
  (16, 128, 10000)
  ↓
BatchNorm → ReLU → Conv1d (k=3, dilation=1)
  ↓
  (16, 128, 10000)
  ↓
Add residual
  ↓
Output: (16, 128, 10000)
```

### Block-by-Block Flow (Alternating Dilation Strategy, alternate=2):

**ResBlock 0: dilation=1 (kernel=9)**
- Input: (16, 128, 10000)
- Output: (16, 128, 10000)
- Receptive field: 15 + (9*1) + 3 = 27 positions

**ResBlock 1: dilation=1 (kernel=9)**
- Input: (16, 128, 10000)
- Output: (16, 128, 10000)
- Receptive field: 27 + (9*1) + 3 = 39 positions
- **Scale Group 0 ends here** (after 2 blocks with d=1)
- Skip collected via group_skip_projections[0]: (16, 128, 10000)

**ResBlock 2: dilation=2 (kernel=9)**
- Input: (16, 128, 10000)
- Output: (16, 128, 10000)
- Receptive field: 39 + (9*2) + 3 = 60 positions

**ResBlock 3: dilation=2 (kernel=9)**
- Input: (16, 128, 10000)
- Output: (16, 128, 10000)
- Receptive field: 60 + (9*2) + 3 = 81 positions
- **Scale Group 1 ends here** (after 2 blocks with d=2)
- Skip collected via group_skip_projections[1]: (16, 128, 10000)

**ResBlock 4: dilation=4 (kernel=9)**
- Input: (16, 128, 10000)
- Output: (16, 128, 10000)
- Receptive field: 81 + (9*4) + 3 = 120 positions

**ResBlock 5: dilation=4 (kernel=9)**
- Input: (16, 128, 10000)
- Output: (16, 128, 10000)
- Receptive field: 120 + (9*4) + 3 = 159 positions
- **Scale Group 2 ends here** (after 2 blocks with d=4)
- Skip collected via group_skip_projections[2]: (16, 128, 10000)

**ResBlock 6: dilation=8 (kernel=9)**
- Input: (16, 128, 10000)
- Output: (16, 128, 10000)
- Receptive field: 159 + (9*8) + 3 = 234 positions

**ResBlock 7: dilation=8 (kernel=9)**
- Input: (16, 128, 10000)
- Output: (16, 128, 10000)
- Receptive field: 234 + (9*8) + 3 = 309 positions
- **Scale Group 3 ends here** (after 2 blocks with d=8)
- Skip collected via group_skip_projections[3]: (16, 128, 10000)

---

## 5. Multi-Scale Feature Fusion

### Skip Features Collection
```
skip_features = [
    initial_skip:              (16, 128, 10000)  # Scale 0: embedding
    group_skip_projections[0]: (16, 128, 10000)  # Scale 1: d=1,1 (blocks 0-1)
    group_skip_projections[1]: (16, 128, 10000)  # Scale 2: d=2,2 (blocks 2-3)
    group_skip_projections[2]: (16, 128, 10000)  # Scale 3: d=4,4 (blocks 4-5)
    group_skip_projections[3]: (16, 128, 10000)  # Scale 4: d=8,8 (blocks 6-7)
]
Total scales: 5
```

### After Concatenation
```
5 scales * 128 channels = 640 channels
Shape: (16, 640, 10000)
Memory: ~410 MB (16 * 640 * 10000 * 4 bytes)
Description: All multi-scale features stacked together
```

### After Bottleneck Fusion
**fusion_reduce (1x1 conv: 640 → 128)**
```
Shape: (16, 128, 10000)
Memory reduction: 410 MB → 82 MB (5x compression)
Description: Compress multi-scale information
```

**fusion_activation (ReLU)**
```
Shape: (16, 128, 10000)
Description: Non-linearity
```

**fusion_expand (1x1 conv: 128 → 128)**
```
Shape: (16, 128, 10000)
Description: Final fused multi-scale features
```

### After Transpose
```
Shape: (16, 10000, 128)
Description: Back to sequence-first format for layer norm
Memory: ~82 MB
```

### After LayerNorm
```
Shape: (16, 10000, 128)
Description: Normalized across embedding dimension
```

### After Dropout (p=0.5)
```
Shape: (16, 10000, 128)
Description: Regularization applied (50% of activations zeroed during training)
Name: fused_skip
```

---

## 6. Central Region Extraction

### Extract Central Region (remove context)
```
Input:  fused_skip = (16, 10000, 128)
        [:, 4500:-4500, :]
Output: central_skip = (16, 1000, 128)
Description: Only the middle 1000 positions (predictions region)
Memory: ~8.2 MB (16 * 1000 * 128 * 4 bytes)
Context removed: 4500 positions on each side are discarded
```

### Also extract from encoder_features
```
Input:  encoder_features = (16, 10000, 128)
        [:, 4500:-4500, :]
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
    'encoder_features': (16, 10000, 128),  # Full sequence features
    'central_features': (16, 1000, 128),   # Central region features
    'skip_features': (16, 10000, 128)      # Fused multi-scale skip (full)
}
```

---

## 9. Prediction Mode Output (via .predict())

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

## Memory Summary

**Peak memory usage (during forward pass with batch_size=16):**

1. Input: ~2.6 MB
2. Embedding: ~82 MB
3. ResBlock processing: ~82 MB (activations reused)
4. Skip concatenation: **~410 MB (peak!)**
5. After bottleneck fusion: ~82 MB
6. Central region: ~8.2 MB
7. Outputs: ~1.1 MB

**Total peak: ~410 MB** (during skip concatenation)
**After fusion: ~82 MB** (5x compression achieved)

**Scaling to batch_size=128:**
- Peak memory: ~3.28 GB (8x larger)
- With mixed precision (FP16/BF16): ~1.64 GB at peak
- Requires significant GPU memory management

---

## Receptive Field Growth

The receptive field grows with each residual block:

| Block | Dilation | Receptive Field | What it captures |
|-------|----------|-----------------|------------------|
| Input conv | - | 15 bp | Immediate neighbors |
| ResBlock 0 | 1 | 27 bp | Local sequence context |
| ResBlock 1 | 1 | 39 bp | Extended local context |
| ResBlock 2 | 2 | 60 bp | Nearby motifs |
| ResBlock 3 | 2 | 81 bp | Splice site consensus regions |
| ResBlock 4 | 4 | 120 bp | Exon/intron boundaries |
| ResBlock 5 | 4 | 159 bp | Branch point regions |
| ResBlock 6 | 8 | 234 bp | Distant regulatory elements |
| ResBlock 7 | 8 | **309 bp** | Long-range context |

**Final receptive field: 309 nucleotides**

This means each output position can "see" ±154 positions around it, sufficient to capture:
- Canonical splice site motifs (GT-AG, ~2-20 bp)
- Branch points (20-50 bp upstream of acceptors)
- Polypyrimidine tracts
- Exonic/intronic splicing enhancers/silencers (ESE/ESS/ISE/ISS)

---

## Key Design Decisions

1. **Multi-scale skip connections**: Capture features at different dilation scales
2. **Alternating dilation pattern (alternate=2)**: Pairs of blocks at each dilation [1,1,2,2,4,4,8,8] provide stable feature extraction at multiple scales
3. **Bottleneck fusion**: Compress 5 scales from 640 → 128 channels (5x memory reduction)
4. **Context removal**: Only predict on central region, use flanking sequence as context
5. **Separate heads**: Independent 1x1 convolutions for classification and regression
6. **Pre-activation ResBlocks**: Better gradient flow and feature reuse
7. **Dropout (0.5)**: Strong regularization to prevent overfitting
8. **Smaller embed_dim (128)**: Reduces memory footprint compared to larger models
9. **Large receptive field (309bp)**: Captures both local motifs and distant regulatory signals
10. **1x1 convolutions for output**: Maintains position independence in predictions
