import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ResBlock(nn.Module):
    """Residual block with pre-activation design (BN-ReLU-Conv)."""
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 15, 
                 stride: int = 1, 
                 dilation: int = 1):
        super().__init__()

        padding = int(1 / 2 * (1 - in_channels + dilation * (kernel_size - 1) - stride + in_channels * stride))

        # Pre-activation: BN → ReLU → Conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        
        # Skip connection (no BN here in pre-activation)
        self.skip_connection = None
        if in_channels != out_channels or stride != 1:
            self.skip_connection = nn.Conv1d(in_channels, out_channels, 1, stride, 0)
    
    def forward(self, x):
        residual = x
        
        # Pre-activation path
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        # Skip connection (just projection, no BN)
        if self.skip_connection is not None:
            residual = self.skip_connection(x)
            
        # Add residual (no final activation in pre-activation design)
        out += residual
        
        return out


class EncoderModule(nn.Module):
    """Encoder module with residual blocks that works directly on one-hot encoded sequences."""
    
    def __init__(self, 
                 embed_dim: int = 256, 
                 num_resblocks: int = 4,
                 dilation_strategy: str = 'exponential',
                 num_classes: int = 3,
                 n_conditions: int = 5,
                 add_output_heads: bool = True,
                 context_len: int = 4500,
                 dropout: float = 0.0):
        """
        Initialize encoder module.
        
        Args:
            embed_dim: Embedding dimension (number of channels)
            num_resblocks: Number of residual blocks
            dilation_strategy: Strategy for dilation rates
            num_classes: Number of splice site classes (3: none, donor, acceptor)
            n_conditions: Number of tissue/timepoint conditions for usage prediction
            add_output_heads: Whether to add classification and usage prediction heads
            context_len: Number of positions on each end to treat as context (removed from output)
            dropout: Dropout rate (default: 0.0)
        """
        super().__init__()
        
        self.n_conditions = n_conditions
        self.add_output_heads = add_output_heads
        self.context_len = context_len
        
        # Initial convolution to project from one-hot (4 channels) to embed_dim
        # Use padding=7 to keep sequence length unchanged (for kernel_size=15)
        self.input_conv = nn.Conv1d(4, embed_dim, kernel_size=15, padding=7)

        # Don't use batchnorm after first layer convolution, somehow it ruins interpretability 
        # self.input_bn = nn.BatchNorm1d(embed_dim)

        # ReLU activation
        self.input_relu = nn.ReLU(inplace=True)
        
        # Calculate dilation rates based on strategy
        dilations = self._get_dilations(num_resblocks, dilation_strategy)

        # Calculate kernel sizes 
        kernel_sizes = self._get_kernel_sizes(num_resblocks)

        # Series of residual blocks with varying dilation
        self.resblocks = nn.ModuleList()
        for i in range(num_resblocks):
            self.resblocks.append(
                ResBlock(embed_dim, embed_dim, kernel_size=kernel_sizes[i], dilation=dilations[i])
            )
        
        # Group blocks by dilation to identify scales
        self.dilation_groups = []
        current_group = [0]
        for i in range(1, num_resblocks):
            if dilations[i] == dilations[i-1]:
                current_group.append(i)
            else:
                self.dilation_groups.append(current_group)
                current_group = [i]
        self.dilation_groups.append(current_group)  # Add final group
        
        # Number of scales = initial embedding + number of dilation groups
        num_scales = len(self.dilation_groups) + 1
        
        # Initial skip projection (1x1 conv on embedding)
        self.initial_skip_proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        
        # Create 1x1 convs for each dilation group
        self.group_skip_projections = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
            for _ in self.dilation_groups
        ])
        
        # Bottleneck fusion to reduce memory
        # Concatenated channels: num_scales * embed_dim
        # Bottleneck: reduce to embed_dim, then back to embed_dim
        bottleneck_dim = embed_dim 
        self.fusion_reduce = nn.Conv1d(num_scales * embed_dim, bottleneck_dim, kernel_size=1)
        self.fusion_activation = nn.ReLU(inplace=True)
        self.fusion_expand = nn.Conv1d(bottleneck_dim, embed_dim, kernel_size=1)
            
        # Output normalization
        self.output_norm = nn.LayerNorm(embed_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Optional output heads
        if add_output_heads:
            # Splice site classification head
            self.splice_classifier = nn.Conv1d(embed_dim, num_classes, kernel_size=1)
            
            # Usage prediction head
            self.usage_predictor = nn.Conv1d(embed_dim, n_conditions * 3, kernel_size=1)

    def _get_kernel_sizes(self, num_blocks: int):
        """
        Get kernel sizes for each residual block.
        
        Strategy: Start with larger kernels for motif detection, 
        decrease as dilation provides larger receptive field.
        
        Args:
            num_blocks: Number of residual blocks
            
        Returns:
            List of kernel sizes
        """
        # Simple strategy: constant kernel size (let dilation do the work)
        # This is often the best approach - simple and effective
        return [9] * num_blocks
        
        # Alternative: Decrease kernel size in later layers
        # if num_blocks <= 4:
        #     return [11, 9, 7, 5][:num_blocks]
        # else:
        #     # Gradual decrease from 11 to 5
        #     step = (11 - 5) / (num_blocks - 1)
        #     return [max(5, int(11 - i * step)) for i in range(num_blocks)]
        

    def _get_dilations(self, num_blocks: int, strategy: str, custom_dilations: Optional[list] = None, alternate: Optional[int] = 4):
        """
        Get dilation rates for each residual block.
        Either predefined strategies or custom_dilations list should be provided. 
        If both are given, custom_dilations takes precedence.
        
        Args:
            num_blocks: Number of residual blocks
            strategy: Dilation strategy
            custom_dilations: Optional list of custom dilation rates
            alternate: Parameter for 'alternating' strategy
            
        Returns:
            List of dilation rates
        """
        if custom_dilations is not None:
            if len(custom_dilations) != num_blocks:
                raise ValueError("Length of custom dilation list must match num_blocks")
            return custom_dilations
        if strategy == 'none':
            return [1] * num_blocks
        elif strategy == 'exponential':
            # [1, 2, 4, 8, 16, ...] but cap at a reasonable value
            return [min(2**i, 32) for i in range(num_blocks)]
        elif strategy == 'linear':
            # [1, 2, 3, 4, ...]
            return [i + 1 for i in range(num_blocks)]
        elif strategy == 'alternating':
            if alternate <= 0:
                raise ValueError("Alternate parameter must be > 0 for 'alternating' strategy")
            return [2**(i // alternate) for i in range(num_blocks)]
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
                    - 'skip_features': Fused skip features (batch_size, seq_len, embed_dim)
            If add_output_heads=False:
                Central region features of shape (batch_size, central_len, embed_dim)
        """
        # Transpose for conv1d: (batch_size, 4, seq_len)
        x = sequences.transpose(1, 2)
        
        # Project from 4 channels to embed_dim
        x = self.input_conv(x)
        # Don't use batchnorm after first layer convolution, somehow ruins interpretability
        # x = self.input_bn(x)
        x = self.input_relu(x)
        
        # Initialize skip from embedding via 1x1 conv
        initial_skip = self.initial_skip_proj(x)
        
        # Collect skip features from all scales
        skip_features = [initial_skip]
        
        # Apply residual blocks and collect features at each scale
        group_idx = 0
        for i, resblock in enumerate(self.resblocks):
            x = resblock(x)
            
            # Check if we're at the end of a dilation group
            if group_idx < len(self.dilation_groups) and i == self.dilation_groups[group_idx][-1]:
                # Project features and add to skip collection
                group_skip = self.group_skip_projections[group_idx](x)
                skip_features.append(group_skip)
                group_idx += 1
        
        # Concatenate all skip features along channel dimension
        # Each has shape (batch, embed_dim, seq_len)
        concatenated_skip = torch.cat(skip_features, dim=1)  # (batch, num_scales*embed_dim, seq_len)
        
        # Bottleneck fusion to reduce memory and fuse multi-scale features
        fused_skip = self.fusion_reduce(concatenated_skip)
        fused_skip = self.fusion_activation(fused_skip)
        fused_skip = self.fusion_expand(fused_skip)  # (batch, embed_dim, seq_len)
        
        # Transpose back: (batch_size, seq_len, embed_dim)
        fused_skip = fused_skip.transpose(1, 2)
        encoder_features = x.transpose(1, 2)
        
        # Output normalization on fused skip
        fused_skip = self.output_norm(fused_skip)
        
        # Apply dropout
        fused_skip = self.dropout(fused_skip)
        
        # Extract central region (remove context from both ends)
        # If context_len = 0, use full sequence
        if self.context_len > 0:
            central_skip = fused_skip[:, self.context_len:-self.context_len, :]
            central_features = encoder_features[:, self.context_len:-self.context_len, :]
        else:
            central_skip = fused_skip
            central_features = encoder_features
        
        # If no output heads, just return central skip features
        if not self.add_output_heads:
            return central_skip
        
        # Transpose central_skip for conv1d: (batch_size, embed_dim, central_len)
        central_skip_conv = central_skip.transpose(1, 2)
        
        # Apply output heads to central skip
        # Output 1: Classify splice sites
        splice_logits = self.splice_classifier(central_skip_conv)  # (batch, num_classes, central_len)
        
        # Output 2: Predict usage statistics
        usage_flat = self.usage_predictor(central_skip_conv)  # (batch, n_conditions*3, central_len)
        
        # Transpose back: (batch, central_len, num_classes/n_conditions*3)
        splice_logits = splice_logits.transpose(1, 2)
        usage_flat = usage_flat.transpose(1, 2)
        
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
            output['skip_features'] = fused_skip  # Fused multi-scale skip features
            
        return output


class SplicevoModel(nn.Module):
    """Simplified model with encoder and dual decoders for splice site and usage prediction."""
    
    def __init__(self, 
                 embed_dim: int = 256, 
                 num_resblocks: int = 4,
                 dilation_strategy: str = 'exponential',
                 alternate: Optional[int] = 4,
                 num_classes: int = 3,
                 n_conditions: int = 5,
                 context_len: int = 4500,
                 dropout: float = 0.3):
        """
        Initialize model with encoder and dual decoders.
        
        Args:
            embed_dim: Embedding dimension (channels)
            num_resblocks: Number of residual blocks in encoder
            dilation_strategy: Dilation strategy for residual blocks
            num_classes: Number of splice site classes (3: none, donor, acceptor)
            n_conditions: Number of tissue/timepoint conditions for usage prediction
            context_len: Number of positions on each end to treat as context
                        For input of length L, predictions are made for positions
                        [context_len : L - context_len]
            dropout: Dropout rate (default: 0.3)
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
            add_output_heads=True,
            context_len=context_len,
            dropout=dropout
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
    
    def predict(self, sequences, batch_size: int = 32):
        """
        Predict splice sites and usage statistics for central region with batching.
        
        Args:
            sequences: One-hot encoded DNA sequences of shape (batch_size, seq_len, 4)
                      or (seq_len, 4) for single sequence
                      Can be numpy array or torch tensor (will be kept on CPU initially)
                      Format: [context_len | central_region | context_len]
            batch_size: Batch size for processing (default: 32)
            
        Returns:
            Dictionary with numpy arrays for central region:
                - 'splice_predictions': Predicted classes (batch_size, central_len)
                - 'splice_probs': Class probabilities (batch_size, central_len, num_classes)
                - 'splice_logits': Raw logits (batch_size, central_len, num_classes)
                - 'usage_predictions': Usage statistics (batch_size, central_len, n_conditions, 3)
                  where last dimension is [alpha, beta, sse]
        """
        import numpy as np
        
        self.eval()
        
        # Convert to tensor if needed, but keep on CPU
        if isinstance(sequences, np.ndarray):
            sequences = torch.from_numpy(sequences).float()
        
        # Handle single sequence input
        single_input = False
        if sequences.ndim == 2:
            sequences = sequences.unsqueeze(0)
            single_input = True
        
        num_sequences = sequences.shape[0]
        device = next(self.parameters()).device
        
        # Initialize lists to collect results (as numpy arrays)
        all_splice_logits = []
        all_splice_probs = []
        all_splice_predictions = []
        all_usage_predictions = []
        
        with torch.no_grad():
            # Process in batches - only transfer one batch at a time to GPU
            for i in range(0, num_sequences, batch_size):
                # Get batch (still on CPU)
                batch_end = min(i + batch_size, num_sequences)
                batch_sequences = sequences[i:batch_end]
                
                # Transfer only this batch to device
                batch_sequences = batch_sequences.to(device)
                
                # Forward pass
                output = self.forward(batch_sequences)
                
                # Process splice site predictions
                splice_logits = output['splice_logits']
                splice_probs = F.softmax(splice_logits, dim=-1)
                splice_predictions = splice_logits.argmax(dim=-1)
                
                # Usage predictions are already in the correct format
                usage_predictions = output['usage_predictions']
                
                # Move results back to CPU and convert to numpy immediately
                # This frees GPU memory for the next batch
                all_splice_logits.append(splice_logits.cpu().numpy())
                all_splice_probs.append(splice_probs.cpu().numpy())
                all_splice_predictions.append(splice_predictions.cpu().numpy())
                all_usage_predictions.append(usage_predictions.cpu().numpy())
                
                # Clear GPU cache after each batch
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Concatenate all batches (numpy arrays)
        splice_logits = np.concatenate(all_splice_logits, axis=0)
        splice_probs = np.concatenate(all_splice_probs, axis=0)
        splice_predictions = np.concatenate(all_splice_predictions, axis=0)
        usage_predictions = np.concatenate(all_usage_predictions, axis=0)
        
        # If single input, remove batch dimension
        if single_input:
            splice_logits = splice_logits[0]
            splice_probs = splice_probs[0]
            splice_predictions = splice_predictions[0]
            usage_predictions = usage_predictions[0]
        
        return {
            'splice_logits': splice_logits,
            'splice_probs': splice_probs,
            'splice_predictions': splice_predictions,
            'usage_predictions': usage_predictions
        }