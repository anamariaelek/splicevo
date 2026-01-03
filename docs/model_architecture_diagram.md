# SplicevoModel Architecture Diagram

## Methods: Model Architecture

### Overview

The Splicevo is a deep convolutional neural network designed for predicting splice site locations and splice site usage across multiple biological conditions. The model processes DNA sequences of fixed length (1900 bp) using a hierarchical feature extraction pipeline with multi-scale skip connections, ultimately generating predictions for a central region of 1000 bp. The architecture combines principles from residual learning with dilated convolutions to achieve an effective receptive field that spans the entire input sequence.

### Input Representation

DNA sequences are represented as one-hot encoded vectors with 4 channels corresponding to nucleotides (A, C, G, T). The input tensor has shape $(B, 1900, 4)$ where $B$ is the batch size and 1900 bp is the total sequence length including flanking context regions. Each sequence is centered around a region of interest (ROI), with 450 bp of context on each side and 1000 bp in the central region where predictions are made.

### Initial Embedding

The one-hot encoded sequences are projected into a high-dimensional embedding space through an initial convolution layer:
- **Layer**: Conv1d(in_channels=4, out_channels=$d_{emb}$, kernel_size=7, padding=3)
- **Output shape**: $(B, 1900, d_{emb})$
- **Activation**: ReLU
- **Default $d_{emb}$**: 128 channels

This initial projection learns task-relevant representations of the nucleotide sequences while maintaining spatial structure through padding that preserves the sequence length.

### Residual Block Architecture

The core of the model consists of 16 residual blocks organized in groups by dilation rate. Each residual block uses a pre-activation design (BatchNorm → ReLU → Conv) following best practices from deep residual networks:

**Single ResBlock Structure:**
1. BatchNorm1d($d_{emb}$)
2. ReLU activation
3. Conv1d(in_channels=$d_{emb}$, out_channels=$d_{emb}$, kernel_size=9, dilation=$d$, stride=1)
4. BatchNorm1d($d_{emb}$)
5. ReLU activation
6. Conv1d(in_channels=$d_{emb}$, out_channels=$d_{emb}$, kernel_size=3, dilation=1, stride=1)
7. Element-wise addition with residual connection

#### Pre-Activation Design Benefits

The pre-activation design (BN-ReLU-Conv ordering) is crucial for training deep networks effectively. In traditional post-activation designs (Conv-BN-ReLU), the skip connection carries an activation function applied to the previous layer, which can lead to gradient bottlenecks. In contrast, the pre-activation approach ensures that:

1. **Direct Gradient Flow**: Gradients flow directly through the skip connection without passing through any non-linearity or normalization. This creates a "short-circuit" path for gradient backpropagation, preventing vanishing gradients in deep networks.

2. **Clean Skip Path**: The skip connection carries the unmodified input $x$, allowing gradients to flow backward unchanged. The output of the block is $f(x) + x$, where $f(x)$ represents the convolution path. This ensures the gradient with respect to input is approximately $\frac{\partial}{\partial x}(f(x) + x) \approx 1 + \frac{\partial f}{\partial x}$, which remains close to 1 even when $\frac{\partial f}{\partial x}$ is small.

3. **Normalization Before Processing**: By applying batch normalization before the convolutions in each residual block, the inputs to the convolutional layers are normalized, which stabilizes training and allows for higher learning rates.

#### Skip Connection Implementation

Skip connections are implemented as follows:

- **Identity Mapping (default)**: When input and output dimensions match (which is always the case in this architecture, as all residual blocks maintain $d_{emb}$ channels), the skip connection is simply the identity operation: $\text{skip} = x$

- **Projection (if needed)**: In cases where dimensions would differ (e.g., between different parts of a network), a learnable 1×1 convolution projects the input to match the output dimensions. However, this is not used in the main residual blocks of SplicevoModel since channel dimensions are preserved throughout.

#### Why This Matters for Deep Networks

With 16 sequential residual blocks, the network could theoretically suffer from vanishing gradients. However, the pre-activation design with clean skip connections ensures that:
- The gradient signal can propagate from the output heads all the way back to the initial layers without significant attenuation
- Each block can learn either to enhance the signal ($f(x) > 0$) or suppress it ($f(x) < 0$), but the gradient backbone remains stable
- The network can effectively use all 16 layers without getting stuck in local minima during training

### Multi-Scale Dilated Convolutions

To efficiently increase the receptive field without excessive downsampling, the 16 residual blocks are organized into 4 groups with progressively increasing dilation rates (alternating dilation strategy):

- **Blocks 0–3**: dilation = 1 (fine-scale features, RF = 85 positions)
- **Blocks 4–7**: dilation = 2 (intermediate-scale features, RF = 245 positions)
- **Blocks 8–11**: dilation = 4 (coarse-scale features, RF = 565 positions)
- **Blocks 12–15**: dilation = 8 (very coarse-scale features, RF = 1205 positions)

The receptive field (RF) for each block $n$ is calculated as:
$$RF_n = RF_{n-1} + 2(k_{conv} - 1) \cdot d + 2(k_{proj} - 1) \cdot d$$

where $k_{conv} = 9$ is the main convolution kernel size, $k_{proj} = 3$ is the projection convolution kernel size, and $d$ is the dilation rate. This design allows the model to capture hierarchical features from local to global scales.

### Multi-Scale Skip Connection Collection

Following the design principle of U-Net and similar architectures, skip connections are collected at each scale:

1. **Initial Skip**: Projection of the initial embedding via 1×1 convolution, shape $(B, 1900, d_{emb})$
2. **Group Skip Connections**: After each dilation group completes, a 1×1 convolution projects the features, yielding 4 additional skip connections of shape $(B, 1900, d_{emb})$

This results in 5 multi-scale feature maps collected from:
- The initial embedding (finest resolution)
- 4 groups of residual blocks at progressively coarser scales

### Feature Fusion

All 5 skip connections are concatenated along the channel dimension:
$$\text{concat\_features} = [\text{skip}_0 \| \text{skip}_1 \| \text{skip}_2 \| \text{skip}_3 \| \text{skip}_4]$$

This produces a tensor of shape $(B, 1900, 5 \cdot d_{emb})$ containing information from 5 different receptive field scales.

#### Why Concatenate Multiple Scales?

The multi-scale skip connections serve as a **feature pyramid** that captures information at different levels of spatial context:
- **Fine-scale features** (from early, low-dilation blocks): Capture local, position-specific details important for precise splice site localization
- **Coarse-scale features** (from late, high-dilation blocks): Capture broader sequence context important for understanding the biological environment around splice sites

By concatenating all scales, the model has access to complementary information: precise local patterns AND broader contextual patterns. This is particularly important for genomic sequences where splice sites follow specific local motifs (e.g., GT-AG dinucleotides) but their usage depends on surrounding regulatory elements (e.g., ESE/ESS sequences located 20-100 bp away).

#### Bottleneck Fusion Module

A bottleneck fusion module then compresses and harmonizes this multi-scale information:

1. **Reduce**: Conv1d($5 \cdot d_{emb} \to d_{emb}$, kernel_size=1) — **5× channel compression**
   - *Purpose*: Compresses the redundancy inherent in 5 concatenated feature maps. Since all 5 skip connections contain overlapping information (all processed the same input), they are highly correlated. The 1×1 convolution learns to fuse and eliminate redundancy.
   - *Information bottleneck principle*: Forces the network to learn the most salient combinations of multi-scale features, acting as a regularizer that prevents overfitting

2. **Activation**: ReLU
   - *Purpose*: Introduces non-linearity to allow complex feature interactions

3. **Expand**: Conv1d($d_{emb} \to d_{emb}$, kernel_size=1) — **maintains representation capacity**
   - *Purpose*: After the reduce step compresses 640 channels to 128, the expand step takes these 128 compressed channels and projects them back to 128 channels (not expanding beyond the input).
   - *Why keep it at 128 and not expand back to 640?* The expand layer is NOT trying to recover the original 640-channel representation. Instead, it's learning a **different 128-dimensional representation** that incorporates the compressed information from all 5 scales. This is the key insight of the bottleneck design:
     - The reduce layer learns **which combinations of the 5 scales are most important** (640 → 128), eliminating redundancy
     - The expand layer then learns **how to best utilize this compressed information** for the downstream tasks (128 → 128), learning feature combinations specifically optimized for splice site prediction and usage estimation
   - *Why not just use the output of the reduce layer?* The bottleneck block with reduce-expand is a standard deep learning pattern because:
     - **Non-linearity**: The ReLU between reduce and expand allows the network to learn non-linear combinations. If we skipped expand and went directly to output heads, the multi-scale fusion would be limited to linear combinations
     - **Learned transformation**: The expand layer can learn task-specific feature transformations that the reduce layer (which focuses on compression) may not discover
     - **Expressivity**: While 128 channels seems small compared to 640, the non-linearity of ReLU means the network has more expressive power than a simple linear projection

4. **LayerNorm**: Normalization across the embedding dimension
   - *Purpose*: Stabilizes the fused representation before it reaches the output heads, enabling more stable gradient flow

5. **Dropout**: Applied at rate $p=0.5$ during training for regularization
   - *Purpose*: Prevents co-adaptation of feature detectors and reduces overfitting, particularly important since the fused features serve as input to both output heads

#### Result

The fused multi-scale features now have shape $(B, 1900, d_{emb})$ and contain both fine and coarse-scale information across the entire sequence, compressed to the same dimensionality as the individual skip connections. This enables:
- **Computational efficiency**: Output heads operate on $(B, 1900, 128)$ instead of $(B, 1900, 640)$ (5× parameter reduction)
- **Regularization**: The bottleneck acts as a compression layer that forces the network to learn robust multi-scale features
- **Unified representation**: Both splice site classification and usage prediction heads receive a single, coherent representation combining all scales

### Central Region Extraction

Since edge regions of the sequence may suffer from boundary effects and the context regions are not of interest for biological predictions, only the central 1000 bp are extracted:
$$\text{central\_features} = \text{fused\_features}[:, 450:-450, :]$$

This produces a tensor of shape $(B, 1000, d_{emb})$.

### Output Heads

Two task-specific prediction heads are applied to the central features:

#### Splice Site Classification Head

A 1×1 convolution predicts logits for splice site classification at each position:
- **Layer**: Conv1d($d_{emb} \to 3$, kernel_size=1)
- **Output shape**: $(B, 1000, 3)$
- **Classes**: 
  - Channel 0: No splice site (background)
  - Channel 1: Donor splice site (GT)
  - Channel 2: Acceptor splice site (AG)
- **Loss**: Cross-entropy loss during training

#### Usage Prediction Head (Regression)

A second 1×1 convolution predicts splice site usage (SSE scores) across multiple biological conditions:
- **Layer**: Conv1d($d_{emb} \to n_{cond}$, kernel_size=1)
- **Output shape**: $(B, 1000, n_{cond})$
- **Activation**: Sigmoid (constrains output to [0, 1])
- **Outputs**: Predicted SSE for each of $n_{cond}$ conditions (e.g., 5 tissues/timepoints)
- **Loss**: Mean squared error or weighted variants during training

#### Usage Classification Head (Optional, for Hybrid Loss)

When using a hybrid loss function, an additional head performs classification of usage levels:
- **Layer**: Conv1d($d_{emb} \to 3 \cdot n_{cond}$, kernel_size=1)
- **Output shape**: $(B, 1000, n_{cond}, 3)$
- **Classes per condition**:
  - Class 0: Zero usage ($\text{SSE} < 0.05$)
  - Class 1: Middle usage ($0.05 \le \text{SSE} \le 0.95$)
  - Class 2: Full usage ($\text{SSE} > 0.95$)
- **Loss**: Cross-entropy loss combined with regression loss

### Forward Pass Summary

The complete forward pass proceeds as follows:

1. Encode one-hot sequences through initial convolution + ReLU
2. Process through 16 residual blocks, collecting multi-scale skip connections
3. Concatenate and fuse all skip connections via bottleneck module
4. Extract central 1000 bp region
5. Apply output heads to generate:
   - Splice site classification logits
   - Splice site usage predictions
   - (Optional) Usage classification logits

### Output Format

The model outputs a dictionary containing:

**Core outputs:**
- `splice_logits`: Shape $(B, 1000, 3)$ — classification logits for splice sites
- `usage_predictions`: Shape $(B, 1000, n_{cond})$ — SSE predictions in $[0, 1]$
- `usage_class_logits` (optional): Shape $(B, 1000, n_{cond}, 3)$ — classification logits for usage levels

**Optional auxiliary outputs (when `return_features=True`):**
- `encoder_features`: Shape $(B, 1900, d_{emb})$ — full-sequence embeddings before central extraction
- `central_features`: Shape $(B, 1000, d_{emb})$ — central region embeddings
- `skip_features`: Shape $(B, 1900, d_{emb})$ — fused multi-scale features before central extraction

### Key Design Principles

1. **Pre-Activation Design**: BN-ReLU-Conv ordering enables better gradient flow in deep networks
2. **Multi-Scale Feature Fusion**: Combines hierarchical features from local to global receptive fields
3. **Efficient Receptive Field Growth**: Dilated convolutions with exponentially increasing dilation rates achieve large RF without spatial downsampling
4. **Context Removal**: Edge regions are excluded from predictions to avoid boundary artifacts
5. **Dual Task Learning**: Joint optimization of splice site classification and usage prediction leverages shared representations
6. **Bottleneck Fusion**: Compresses multi-scale information before output heads for computational efficiency and information bottleneck principle

---

## High-Level Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SPLICEVO MODEL ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────────┘

                            INPUT: One-hot DNA Sequences
                                  (B, 1900, 4)
                                       │
                                       ▼
                    ┌───────────────────────────────────┐
                    │  Initial Convolution (4 → 128)    │
                    │  kernel=7, padding=3              │
                    │  Output: (B, 1900, 128)           │
                    └───────────────────────────────────┘
                                       │
                                       ▼
                    ┌───────────────────────────────────┐
                    │        ReLU Activation            │
                    └───────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │ Initial Skip     │                  │
                    │ (1x1 conv)       │                  │
                    │ (B, 1900, 128)   │                  │
                    └──────────┬───────┘                  │
                               │                          │
                               │                    ┌─────▼─────────────┐
                               │                    │  Multi-Scale      │
                               │                    │  Residual Blocks  │
                               │                    │  (16 blocks)      │
                               │                    └─────┬─────────────┘
```

## Residual Block Processing with Multi-Scale Skip Collection

```
                    ┌──────────────────────────────────────────┐
                    │  RESIDUAL BLOCK PROCESSING               │
                    │  (16 blocks grouped by dilation)         │
                    └──────────────────────────────────────────┘

    ┌─────────────────────────┐  ┌──────────────────────┐
    │ Group 0: 4 blocks       │  │ Group 1: 4 blocks    │
    │ dilation = 1            │  │ dilation = 2         │
    │ (Blocks 0-3)            │  │ (Blocks 4-7)         │
    │ RF: 55 positions        │  │ RF: 135 positions    │
    └────────┬────────────────┘  └──────────┬───────────┘
             │ Skip to fusion       │ Skip to fusion
             │ (B, 1900, 128)       │ (B, 1900, 128)
             │                      │
             │  ┌─────────────────────────────────────┐
             │  │ Group 2: 4 blocks                   │
             │  │ dilation = 4                        │
             │  │ (Blocks 8-11)                       │
             │  │ RF: 295 positions                   │
             │  └────────┬────────────────────────────┘
             │           │ Skip to fusion
             │           │ (B, 1900, 128)
             │           │
             │           │  ┌──────────────────────────────┐
             │           │  │ Group 3: 4 blocks            │
             │           │  │ dilation = 8                 │
             │           │  │ (Blocks 12-15)               │
             │           │  │ RF: 615 positions            │
             │           │  └────────┬─────────────────────┘
             │           │           │ Skip to fusion
             │           │           │ (B, 1900, 128)
             │           │           │
             └───────────┼───────────┼──────────┐
                         │           │          │
                         ▼           ▼          ▼
```

## Multi-Scale Feature Fusion

```
                    ┌─────────────────────────────────────┐
                    │  MULTI-SCALE SKIP COLLECTION        │
                    └─────────────────────────────────────┘

    Initial Skip        Group 0 Skip       Group 1 Skip       Group 2 Skip       Group 3 Skip
    (B, 1900, 128)      (B, 1900, 128)     (B, 1900, 128)     (B, 1900, 128)     (B, 1900, 128)
         │                    │                   │                   │                   │
         └────────────────────┼───────────────────┼───────────────────┼───────────────────┘
                              │ (All 5 scales)
                              ▼
                    ┌─────────────────────────────────────┐
                    │  Concatenation                      │
                    │  5 * 128 = 640 channels            │
                    │  Output: (B, 1900, 640)            │
                    └─────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────────────────────┐
                    │  Bottleneck Fusion (1x1 conv)       │
                    │  640 → 128 channels                 │
                    │  Output: (B, 1900, 128)            │
                    └─────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────────────────────┐
                    │  ReLU + Expand (1x1 conv)           │
                    │  Output: (B, 1900, 128)            │
                    └─────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────────────────────┐
                    │  LayerNorm + Dropout (p=0.5)        │
                    │  Output: (B, 1900, 128)            │
                    │  Name: fused_skip                  │
                    └─────────────────────────────────────┘
```

## Central Region Extraction & Output Heads

```
                    ┌─────────────────────────────────────┐
                    │  CENTRAL REGION EXTRACTION          │
                    │  Remove context (450 bp on each end)│
                    │  Extract middle 1000 positions      │
                    │  fused_skip[:, 450:-450, :]        │
                    └─────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────────────────────┐
                    │  Central Features                   │
                    │  (B, 1000, 128)                    │
                    └─────────────────────────────────────┘
                              │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
        ┌──────────────────────────┐     ┌──────────────────────────┐
        │  SPLICE SITE CLASSIFIER  │     │  USAGE PREDICTOR         │
        │  (1x1 conv: 128 → 3)     │     │  (1x1 conv: 128 → 5)    │
        │                          │     │                          │
        │  Output: (B, 1000, 3)    │     │  Output: (B, 1000, 5)   │
        │  Classes:                │     │  Outputs: [SSE for 5    │
        │  - Channel 0: none       │     │   tissue/timepoint cond] │
        │  - Channel 1: donor      │     │                          │
        │  - Channel 2: acceptor   │     │  After Sigmoid:          │
        │                          │     │  usage_predictions       │
        │  Name: splice_logits     │     │  Range: [0, 1]          │
        └──────────────────────────┘     └──────────────────────────┘
                    │                                   │
                    │           ┌───────────────────────┘
                    │           │
                    │           ▼
                    │    ┌──────────────────────────────┐
                    │    │  USAGE CLASSIFIER (Optional)  │
                    │    │  (if hybrid loss)             │
                    │    │  (1x1 conv: 128 → 15)        │
                    │    │                              │
                    │    │  Output: (B, 1000, 5, 3)    │
                    │    │  3 classes per condition:     │
                    │    │  - is_zero (< 0.05)          │
                    │    │  - is_one (> 0.95)           │
                    │    │  - is_middle (0.05-0.95)     │
                    │    │                              │
                    │    │  Name: usage_class_logits   │
                    │    └──────────────────────────────┘
                    │           │
                    └───────────┼───────────────────────┐
                                │                       │
                                ▼                       ▼
                    ┌─────────────────────────┐  ┌─────────────────────────┐
                    │   OUTPUT DICTIONARY     │  │  WITH RETURN_FEATURES=T │
                    │   (core outputs)        │  │  (additional outputs)   │
                    │                         │  │                         │
                    │  'splice_logits':      │  │  'encoder_features':   │
                    │    (B, 1000, 3)        │  │    (B, 1900, 128)      │
                    │                         │  │                         │
                    │  'usage_predictions':  │  │  'central_features':   │
                    │    (B, 1000, 5)        │  │    (B, 1000, 128)      │
                    │                         │  │                         │
                    │  ['usage_class_logits' │  │  'skip_features':      │
                    │   if hybrid loss]      │  │    (B, 1900, 128)      │
                    └─────────────────────────┘  └─────────────────────────┘
```

## Detailed ResBlock Architecture

```
┌────────────────────────────────────────────────────┐
│               RESIDUAL BLOCK                       │
│                                                    │
│  Input: x                                          │
│  Channels: embed_dim (128)                         │
│  Sequence length: 1900 (or 1000 in central)       │
│                                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │  Path 1: Main Processing (Pre-activation)   │  │
│  │                                              │  │
│  │  Input (residual = x)                        │  │
│  │    ▼                                          │  │
│  │  BatchNorm1d(embed_dim)                      │  │
│  │    ▼                                          │  │
│  │  ReLU                                        │  │
│  │    ▼                                          │  │
│  │  Conv1d(embed_dim, embed_dim,                │  │
│  │          kernel=9, dilation=d)               │  │
│  │    ▼                                          │  │
│  │  BatchNorm1d(embed_dim)                      │  │
│  │    ▼                                          │  │
│  │  ReLU                                        │  │
│  │    ▼                                          │  │
│  │  Conv1d(embed_dim, embed_dim,                │  │
│  │          kernel=3, dilation=1)               │  │
│  │    ▼                                          │  │
│  │  out                                         │  │
│  └──────────────────────────────────────────────┘  │
│           │                                         │
│           │        ┌─────────────────────┐         │
│           │        │  Path 2: Skip       │         │
│           │        │  (Identity by       │         │
│           │        │   default, or       │         │
│           │        │   1x1 conv if       │         │
│           │        │   channels differ)  │         │
│           │        │  residual           │         │
│           │        └──────────┬──────────┘         │
│           │                   │                     │
│           └───────────────────┼──────────────────┐  │
│                               │                  │  │
│                               ▼                  ▼  │
│                           Element-wise Add        │
│                               ▼                  │
│                           Output                 │
└────────────────────────────────────────────────────┘
```

## Dilation Strategy for Alternating Groups

```
ALTERNATING DILATION STRATEGY (alternate=4):

Blocks 0-3:    dilation = 1  (kernel=9)
  ├─ Block 0: RF = 15 + 16 + 4 = 25 positions
  ├─ Block 1: RF = 25 + 16 + 4 = 45 positions
  ├─ Block 2: RF = 45 + 16 + 4 = 65 positions
  └─ Block 3: RF = 65 + 16 + 4 = 85 positions [SCALE 1 END]

Blocks 4-7:    dilation = 2  (kernel=9)
  ├─ Block 4: RF = 85 + 32 + 8 = 125 positions
  ├─ Block 5: RF = 125 + 32 + 8 = 165 positions
  ├─ Block 6: RF = 165 + 32 + 8 = 205 positions
  └─ Block 7: RF = 205 + 32 + 8 = 245 positions [SCALE 2 END]

Blocks 8-11:   dilation = 4  (kernel=9)
  ├─ Block 8:  RF = 245 + 64 + 16 = 325 positions
  ├─ Block 9:  RF = 325 + 64 + 16 = 405 positions
  ├─ Block 10: RF = 405 + 64 + 16 = 485 positions
  └─ Block 11: RF = 485 + 64 + 16 = 565 positions [SCALE 3 END]

Blocks 12-15:  dilation = 8  (kernel=9)
  ├─ Block 12: RF = 565 + 128 + 32 = 725 positions
  ├─ Block 13: RF = 725 + 128 + 32 = 885 positions
  ├─ Block 14: RF = 885 + 128 + 32 = 1045 positions
  └─ Block 15: RF = 1045 + 128 + 32 = 1205 positions [SCALE 4 END]

Final Receptive Field: ~1205 positions (covers entire sequence + context)

RF Calculation per block:
  RF(n) = RF(n-1) + 2 * (kernel_size - 1) * dilation
        = RF(n-1) + 2 * (9 - 1) * d
        = RF(n-1) + 16 * d
```

## Data Flow Summary

```
INPUT FLOW:
  One-hot sequences (B, 1900, 4)
         ↓
  Initial Conv (4 → 128)
         ↓
  16 Residual Blocks (varying dilations)
         ↓
  Multi-scale skip connections collected
         ↓
  Bottleneck fusion of 5 scales
         ↓
  Full sequence features (B, 1900, 128)
         ↓
  Extract central region (B, 1000, 128)
         ↓
  Output heads (classification + regression)


OUTPUT:
  ┌─ splice_logits: (B, 1000, 3)
  ├─ usage_predictions: (B, 1000, 5)
  ├─ usage_class_logits: (B, 1000, 5, 3) [if hybrid]
  └─ Optional: encoder_features, central_features, skip_features
```

## Key Design Features

1. **Pre-Activation ResBlocks**: BN-ReLU-Conv design for better training dynamics
2. **Multi-Scale Skip Collection**: Captures features at different receptive field scales
3. **Alternating Dilation Strategy**: Efficiently increases receptive field
4. **Context Removal**: 450 bp context on each end ensures edge effects don't affect predictions
5. **Bottleneck Fusion**: 5× compression of multi-scale features before output heads
6. **Dual Output Heads**: 
   - Splice site classification (3 classes)
   - Usage prediction (5 conditions, continuous SSE)
   - Optional hybrid classification for usage

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | 256 | Embedding dimension throughout |
| `num_resblocks` | 16 | Total residual blocks |
| `num_classes` | 3 | Splice site classes (none, donor, acceptor) |
| `n_conditions` | 5 | Tissue/timepoint conditions for usage |
| `seq_len` | 1900 | Full sequence length |
| `central_len` | 1000 | Central region for predictions |
| `context_len` | 450 | Context removed from each end |
| `kernel_size` | 9 | Kernel size for residual convolutions |
| `dilation_strategy` | 'alternating' | How to assign dilation rates |
| `alternate` | 4 | Number of blocks per dilation level |
| `dropout` | 0.5 | Dropout rate after fusion |

