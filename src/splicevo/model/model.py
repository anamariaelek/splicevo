import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ResBlock(nn.Module):
    """Residual block with convolution, batch normalization, and ReLU."""
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 9, 
                 stride: int = 1, 
                 dilation: int = 1):
        """
        Initialize residual block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels  
            kernel_size: Convolution kernel size (default: 9)
            stride: Convolution stride (default: 1)
            dilation: Dilation rate for dilated convolutions (default: 1)
        """
        super().__init__()

        # Calculate padding to maintain sequence length
        # https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
        padding = int(1 / 2 * (1 - in_channels + dilation * (kernel_size - 1) - stride + in_channels * stride))

        # First conv block
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Second conv block (no dilation, smaller kernel)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.skip_connection = None
        if in_channels != out_channels or stride != 1:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, 0),
                nn.BatchNorm1d(out_channels)
            )
            
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """Forward pass through residual block."""
        residual = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        if self.skip_connection is not None:
            residual = self.skip_connection(x)
            
        # Add residual and apply ReLU
        out += residual
        out = self.relu2(out)
        
        return out


class EncoderModule(nn.Module):
    """Encoder module with residual blocks that works directly on one-hot encoded sequences."""
    
    def __init__(self, 
                 embed_dim: int = 256, 
                 num_resblocks: int = 4,
                 dilation_strategy: str = 'exponential',
                 num_classes: int = 3,
                 n_conditions: int = 5,
                 dropout: float = 0.1,
                 add_output_heads: bool = True,
                 context_len: int = 4500):
        """
        Initialize encoder module.
        
        Args:
            embed_dim: Embedding dimension (number of channels)
            num_resblocks: Number of residual blocks
            dilation_strategy: Strategy for dilation rates
            num_classes: Number of splice site classes (3: none, donor, acceptor)
            n_conditions: Number of tissue/timepoint conditions for usage prediction
            dropout: Dropout rate
            add_output_heads: Whether to add classification and usage prediction heads
            context_len: Number of positions on each end to treat as context (removed from output)
        """
        super().__init__()
        
        self.n_conditions = n_conditions
        self.add_output_heads = add_output_heads
        self.context_len = context_len
        
        # Initial convolution to project from one-hot (4 channels) to embed_dim
        self.input_conv = nn.Conv1d(4, embed_dim, kernel_size=1)
        self.input_bn = nn.BatchNorm1d(embed_dim)
        self.input_relu = nn.ReLU(inplace=True)
        
        # Calculate dilation rates based on strategy
        dilations = self._get_dilations(num_resblocks, dilation_strategy)
        
        # Series of residual blocks with varying dilation
        self.resblocks = nn.ModuleList()
        for i in range(num_resblocks):
            self.resblocks.append(
                ResBlock(embed_dim, embed_dim, dilation=dilations[i])
            )
            
        # Output normalization
        self.output_norm = nn.LayerNorm(embed_dim)
        
        # Optional output heads
        if add_output_heads:
            # Decoder 1: Splice site classification head
            # Minimal: just a linear projection from encoder features
            self.splice_classifier = nn.Linear(embed_dim, num_classes)
            
            # Decoder 2: Usage prediction head
            # Minimal: just a linear projection from encoder features
            self.usage_predictor = nn.Linear(embed_dim, n_conditions * 3)
    
    def _get_dilations(self, num_blocks: int, strategy: str):
        """
        Get dilation rates for each residual block.
        
        Args:
            num_blocks: Number of residual blocks
            strategy: Dilation strategy
            
        Returns:
            List of dilation rates
        """
        if strategy == 'none':
            return [1] * num_blocks
        elif strategy == 'exponential':
            # [1, 2, 4, 8, 16, ...] but cap at a reasonable value
            return [min(2**i, 32) for i in range(num_blocks)]
        elif strategy == 'linear':
            # [1, 2, 3, 4, ...]
            return [i + 1 for i in range(num_blocks)]
        elif strategy == 'alternating':
            # [1, 1, 2, 2, 4, 4, 8, 8, ...]
            return [2**(i // 2) for i in range(num_blocks)]
        elif isinstance(strategy, list):
            if len(strategy) != num_blocks:
                raise ValueError("Length of custom dilation list must match num_blocks")
            return strategy
        else:
            raise ValueError(f"Unknown dilation strategy: {strategy}")
        
    def forward(self, sequences, return_features: bool = False):
        """
        Forward pass through encoder.
        
        Args:
            sequences: One-hot encoded DNA sequences of shape (batch_size, seq_len, 4)
                      Expected format: [context_len | central_region | context_len]
            return_features: Whether to return intermediate features
            
        Returns:
            If add_output_heads=True:
                Dictionary with predictions only for the central region:
                    - 'splice_logits': (batch_size, central_len, num_classes)
                    - 'usage_predictions': (batch_size, central_len, n_conditions, 3)
                If return_features=True, also includes:
                    - 'encoder_features': Full features (batch_size, seq_len, embed_dim)
                    - 'central_features': Central region features (batch_size, central_len, embed_dim)
            If add_output_heads=False:
                Central region features of shape (batch_size, central_len, embed_dim)
        """
        # Transpose for conv1d: (batch_size, 4, seq_len)
        x = sequences.transpose(1, 2)
        
        # Project from 4 channels to embed_dim
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = self.input_relu(x)
        
        # Apply residual blocks
        for resblock in self.resblocks:
            x = resblock(x)
            
        # Transpose back: (batch_size, seq_len, embed_dim)
        x = x.transpose(1, 2)
        
        # Output normalization
        encoder_features = self.output_norm(x)
        
        # Extract central region (remove context from both ends)
        # If context_len = 0, use full sequence
        if self.context_len > 0:
            central_features = encoder_features[:, self.context_len:-self.context_len, :]
        else:
            central_features = encoder_features
        
        # If no output heads, just return central features
        if not self.add_output_heads:
            return central_features
        
        # Apply output heads to central region only
        # Decoder 1: Classify splice sites
        splice_logits = self.splice_classifier(central_features)
        
        # Decoder 2: Predict usage statistics
        usage_flat = self.usage_predictor(central_features)
        
        # Reshape usage predictions: (batch, central_len, n_conditions, 3)
        batch_size, central_len, _ = usage_flat.shape
        usage_predictions = usage_flat.view(batch_size, central_len, self.n_conditions, 3)
        
        output = {
            'splice_logits': splice_logits,
            'usage_predictions': usage_predictions
        }
        
        if return_features:
            output['encoder_features'] = encoder_features  # Full sequence features
            output['central_features'] = central_features  # Central region only
            
        return output


class SplicevoModel(nn.Module):
    """Simplified model with encoder and dual decoders for splice site and usage prediction."""
    
    def __init__(self, 
                 embed_dim: int = 256, 
                 num_resblocks: int = 4,
                 dilation_strategy: str = 'exponential',
                 num_classes: int = 3,
                 n_conditions: int = 5,
                 dropout: float = 0.1,
                 context_len: int = 4500):
        """
        Initialize model with encoder and dual decoders.
        
        Args:
            embed_dim: Embedding dimension (channels)
            num_resblocks: Number of residual blocks in encoder
            dilation_strategy: Dilation strategy for residual blocks
            num_classes: Number of splice site classes (3: none, donor, acceptor)
            n_conditions: Number of tissue/timepoint conditions for usage prediction
            dropout: Dropout rate
            context_len: Number of positions on each end to treat as context
                        For input of length L, predictions are made for positions
                        [context_len : L - context_len]
        """
        super().__init__()
        
        self.context_len = context_len
        
        # Encoder module with output heads
        self.encoder = EncoderModule(
            embed_dim=embed_dim,
            num_resblocks=num_resblocks,
            dilation_strategy=dilation_strategy,
            num_classes=num_classes,
            n_conditions=n_conditions,
            dropout=dropout,
            add_output_heads=True,
            context_len=context_len
        )
        
    def forward(self, sequences, return_features: bool = False):
        """
        Forward pass through the model.
        
        Args:
            sequences: One-hot encoded DNA sequences of shape (batch_size, seq_len, 4)
                      Format: [context_len | central_region | context_len]
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary with predictions for central region only:
                - 'splice_logits': (batch_size, central_len, num_classes)
                  where central_len = seq_len - 2 * context_len
                - 'usage_predictions': (batch_size, central_len, n_conditions, 3)
                  where the last dimension is [alpha, beta, sse]
            If return_features=True, also includes full and central features
        """
        return self.encoder(sequences, return_features=return_features)
    
    def predict(self, sequences, splice_threshold: float = 0.5):
        """
        Predict splice sites and usage statistics for central region.
        
        Args:
            sequences: One-hot encoded DNA sequences of shape (batch_size, seq_len, 4)
                      Format: [context_len | central_region | context_len]
            splice_threshold: Probability threshold for splice site prediction
            
        Returns:
            Dictionary with predictions for central region:
                - 'splice_predictions': Predicted classes (batch_size, central_len)
                - 'splice_probabilities': Class probabilities (batch_size, central_len, num_classes)
                - 'usage_predictions': Usage statistics (batch_size, central_len, n_conditions, 3)
                  where last dimension is [alpha, beta, sse]
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(sequences)
            
            # Process splice site predictions
            splice_logits = output['splice_logits']
            splice_probabilities = F.softmax(splice_logits, dim=-1)
            splice_predictions = splice_logits.argmax(dim=-1)
            
            # Usage predictions are already in the correct format
            usage_predictions = output['usage_predictions']
            
            return {
                'splice_predictions': splice_predictions,
                'splice_probabilities': splice_probabilities,
                'usage_predictions': usage_predictions
            }