# Splice Site Prediction Model Architecture

This module implements a deep learning model for splice site prediction that combines **Residual Convolutional Networks** and **Transformer** architectures to capture both local sequence patterns and long-range dependencies in genomic sequences.

## Architecture Overview

The model architecture consists of an encoder module that processes the input sequence through a series of residual blocks with varying dilation rates, allowing the network to learn features at multiple scales. This is followed by a transformer module that captures long-range dependencies through self-attention mechanisms. The combination of these two architectures enables the model to effectively learn complex patterns in the data.

## Core Classes

### EncoderModule

The `EncoderModule` class constructs an encoder with a series of residual blocks (`ResBlock`) with two dilated 1D convolutional layers and batch normalization.

```python
Input (batch, channels, seq_len)
    ↓
Conv1d(kernel=9, dilation=d) + BatchNorm + ReLU
    ↓
Conv1d(kernel=3, dilation=1) + BatchNorm
    ↓
Add residual connection
    ↓
ReLU
    ↓
Output (batch, channels, seq_len)
```

Each `ResBlock` can potentially use different dilation rates in the first convolutional layer - this allows the model to capture features at multiple scales.  

```python
from splicevo.model import EncoderModule

# 1. No dilation (standard residual network)
# Dilations: [1, 1, 1, 1]
encoder = EncoderModule(
    embed_dim=256,
    num_resblocks=4,
    dilation_strategy='none'
)

# 2. Exponential dilation - fast growth, good for long-range dependencies
# Dilations: [1, 2, 4, 8, 16, 32]
encoder = EncoderModule(
    embed_dim=256,
    num_resblocks=6,
    dilation_strategy='exponential'
)

# 3. Linear dilation - steady growth
# Dilations: [1, 2, 3, 4]
encoder = EncoderModule(
    embed_dim=256,
    num_resblocks=4,
    dilation_strategy='linear'
)

# 4. Alternating dilation - good for hierarchical features 
# Dilations: [1, 1, 2, 2, 4, 4, 8, 8]
encoder = EncoderModule(
    embed_dim=256,
    num_resblocks=8,
    dilation_strategy='alternating'
)

# 5. Custom dilation - user-defined list of dilations
# e.g. SpliceTransformer: [1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10, 25, 25, 25, 25]
encoder = EncoderModule(
    embed_dim=256,
    num_resblocks=16,
    dilation_strategy=[1, 1, 1, 1, 4, 4, 4, 4, 10, 10, 10, 10, 25, 25, 25, 25]
)
```
