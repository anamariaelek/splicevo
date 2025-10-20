# Splice Site Prediction Model Architecture

This module implements a deep learning model for splice site prediction that combines **Residual Convolutional Networks** and **Transformer** architectures to capture both local sequence patterns and long-range dependencies in genomic sequences.

## Architecture Overview

The model architecture consists of an encoder module that processes the input sequence through a series of residual blocks with varying dilation rates, allowing the network to learn features at multiple scales. This is followed by a transformer module that captures long-range dependencies through self-attention mechanisms. The combination of these two architectures enables the model to effectively learn complex patterns in the data.

## Core Classes

### EncoderModule

The `EncoderModule` class constructs an encoder with a series of residual blocks (`ResBlock`) with two dilated 1D convolutional layers and batch normalization. 

#### ResBlock

```
Input (batch, channels, seq_len)
    ↓
Batch Normalization
    ↓
ReLU activation
    ↓
Conv1d(kernel=15, dilation=d)
    ↓
Batch Normalization
    ↓
ReLU activation
    ↓
Conv1d(kernel=3, dilation=1)
    ↓
Add residual connection
    ↓
ReLU activation
```

This is inspired by ResNet design, but it uses a pre-activation design (BatchNorm → ReLU → Conv, as opposed to Conv → BatchNorm → ReLU), as in SpliceTransformer. It has been shown that this approach can lead to better gradient flow and may improve training stability. It performs better on our task, as indicated by faster convergence speed and lower validation loss.

Each `ResBlock` can potentially use different dilation rates in the first convolutional layer - this allows the model to capture features at multiple scales. We use kernel size 15 to have a receptive field large enough to capture relevant sequence motifs. For dilation, we reuse strategy from SpliceTransformer, which employs increasing dilation rates in 16 stacked  `ResBlock`s, to capture long-range dependencies effectively: `[1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10, 20, 20, 20, 20]`. With kernel size $k$ and dilation $d$, the effective receptive field is $d \times (k - 1) + 1$. We use $k=15$ and $d \in [1, 20]$, resulting in receptive fields from 15 to 281 positions.

