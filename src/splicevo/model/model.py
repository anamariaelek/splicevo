import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer with residual connection."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Embedding dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout rate for attention weights
        """
        super().__init__()
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Scaling factor for attention scores
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Attention dropout
        self.attn_dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass of multi-head attention with residual connection.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Save residual connection
        residual = x
        
        # Linear projections and reshape for multi-head attention
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, num_heads, head_dim)
        Q = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for multi-head computation: (batch_size, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention weights to values: (batch_size, num_heads, seq_len, head_dim)
        attn_out = torch.matmul(attn_weights, V)
        
        # Transpose back: (batch_size, seq_len, num_heads, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous()
        
        # Reshape: (batch_size, seq_len, embed_dim)
        attn_out = attn_out.view(batch_size, seq_len, embed_dim)
        
        # Output projection
        attn_out = self.out_proj(attn_out)
        
        # Residual connection and layer normalization
        out = self.norm(attn_out + residual)
        
        # Apply output dropout
        out = self.dropout(out)
        
        return out


class TransformerModule(nn.Module):
    """Transformer module with multi-head self-attention."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        """
        Initialize transformer module.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads (embed_dim must be divisible by num_heads)
            dropout: Dropout rate
        """
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
    
    def forward(self, x):
        """
        Forward pass through transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        return self.attention(x)


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
                 embed_dim: int = 128, 
                 num_resblocks: int = 4,
                 dilation_strategy: str = 'exponential',
                 alternate: Optional[int] = 4,
                 num_classes: int = 3,
                 n_conditions: int = 5,
                 context_len: int = 450,
                 dropout: float = 0.0,
                 usage_loss_type: str = 'weighted_mse',
                 n_species: int = 1,
                 species_names: Optional[list] = None
        ):
        """
        Initialize encoder module (convolutional encoder with multi-scale fusion).
        
        Args:
            embed_dim: Embedding dimension (number of channels)
            num_resblocks: Number of residual blocks
            dilation_strategy: Strategy for dilation rates
            num_classes: Number of splice site classes (3: none, donor, acceptor)
            n_conditions: Number of tissue/timepoint conditions for usage prediction
            context_len: Number of positions on each end to treat as context (removed from output)
            dropout: Dropout rate (default: 0.0)
            usage_loss_type: Type of usage loss ('mse', 'weighted_mse', 'hybrid')
            n_species: Number of species (for tracking)
            species_names: Optional list of species names for tracking
        """
        super().__init__()
        
        self.n_conditions = n_conditions
        self.context_len = context_len
        self.usage_loss_type = usage_loss_type
        self.n_species = n_species
        self.species_names = species_names or [f"species_{i}" for i in range(n_species)]
        
        # Initial convolution to project from one-hot (4 channels) to embed_dim
        self.initial_conv = nn.Conv1d(
            in_channels=4,
            out_channels=embed_dim,
            kernel_size=7,
            padding=3
        )

        # Don't use batchnorm after first layer convolution, somehow it ruins interpretability 
        # self.input_bn = nn.BatchNorm1d(embed_dim)

        # ReLU activation
        self.input_relu = nn.ReLU(inplace=True)
        
        # Calculate dilation rates based on strategy
        dilations = self._get_dilations(num_resblocks, dilation_strategy, alternate=alternate)

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
        self.dilation_groups.append(current_group)
        
        # Number of scales = initial embedding + number of dilation groups
        num_scales = len(self.dilation_groups) + 1
        
        # Initial skip projection (1x1 conv on embedding)
        self.initial_skip_proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        
        # Create 1x1 convs for each dilation group
        self.group_skip_projections = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
            for _ in self.dilation_groups
        ])
        
        # Bottleneck fusion
        bottleneck_dim = embed_dim 
        self.fusion_reduce = nn.Conv1d(num_scales * embed_dim, bottleneck_dim, kernel_size=1)
        self.fusion_activation = nn.ReLU(inplace=True)
        self.fusion_expand = nn.Conv1d(bottleneck_dim, embed_dim, kernel_size=1)
            
        # Output normalization
        self.output_norm = nn.LayerNorm(embed_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

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
        
    def forward(self, sequences, species_ids=None, return_features: bool = False):
        """
        Forward pass through encoder.
        
        Args:
            sequences: One-hot encoded DNA sequences of shape (batch_size, seq_len, 4)
                      Expected format: [context_len | central_region | context_len]
            species_ids: Species IDs (batch,) - integer species identifiers
                        Used to select appropriate species-specific heads
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary with features for the full sequence:
                - 'features': Full sequence features (batch_size, seq_len, embed_dim)
                If return_features=True, also includes:
                    - 'encoder_features': Encoder features (batch_size, seq_len, embed_dim)
                    - 'skip_features': Fused skip features (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = sequences.shape
        
        # Conv1d expects (batch, channels, length)
        x = sequences.transpose(1, 2)
        
        # Project from 4 channels to embed_dim
        x = self.initial_conv(x)
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
        concatenated_skip = torch.cat(skip_features, dim=1)
        
        # Bottleneck fusion to reduce memory and fuse multi-scale features
        fused_skip = self.fusion_reduce(concatenated_skip)
        fused_skip = self.fusion_activation(fused_skip)
        fused_skip = self.fusion_expand(fused_skip)
        
        # Transpose back: (batch_size, seq_len, embed_dim)
        fused_skip = fused_skip.transpose(1, 2)
        encoder_features = x.transpose(1, 2)
        
        # Output normalization on fused skip
        fused_skip = self.output_norm(fused_skip)
        
        # Apply dropout
        fused_skip = self.dropout(fused_skip)
        
        # Return full sequence features (central extraction will happen in model)
        output = {
            'features': fused_skip,        # (batch_size, seq_len, embed_dim)
        }
        
        if return_features:
            output['encoder_features'] = encoder_features
            output['skip_features'] = fused_skip
            
        return output


class SplicevoModel(nn.Module):
    """Model with encoder, transformer, and output heads for splice site and usage prediction."""
    
    def __init__(self, 
                 embed_dim: int = 256, 
                 num_resblocks: int = 4,
                 dilation_strategy: str = 'exponential',
                 alternate: Optional[int] = 4,
                 num_classes: int = 3,
                 n_conditions: int = 5,
                 context_len: int = 450,
                 num_heads: int = 8,
                 dropout: float = 0.3,
                 usage_loss_type: str = 'weighted_mse',
                 n_species: int = 1,
                 species_names: Optional[list] = None
        ):
        """
        Initialize model with encoder, transformer, and output heads.
        
        Args:
            embed_dim: Embedding dimension
            num_resblocks: Number of residual blocks in encoder
            dilation_strategy: Dilation strategy for residual blocks
            num_classes: Number of splice site classes
            n_conditions: Number of conditions (tissues/timepoints) for SSE prediction
            context_len: Number of positions on each end as context
            num_heads: Number of attention heads in transformer
            dropout: Dropout rate
            usage_loss_type: Type of usage loss ('mse', 'weighted_mse', 'hybrid')
            n_species: Number of species (creates separate output heads per species)
            species_names: Optional list of species names
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.context_len = context_len
        self.usage_loss_type = usage_loss_type
        self.n_species = n_species
        self.species_names = species_names or [f"species_{i}" for i in range(n_species)]
        self.output_type = 'splice'  # For gReLU compatibility
        self.prediction_transform = None  # For gReLU compatibility
        
        # Encoder module (convolutional encoder with multi-scale fusion)
        self.encoder = EncoderModule(
            embed_dim=embed_dim,
            num_resblocks=num_resblocks,
            dilation_strategy=dilation_strategy,
            alternate=alternate,
            num_classes=num_classes,
            n_conditions=n_conditions,
            context_len=context_len,
            dropout=dropout,
            usage_loss_type=usage_loss_type,
            n_species=n_species,
            species_names=species_names
        )
        
        # Transformer module (multi-head self-attention)
        self.transformer = TransformerModule(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        
        # Output heads for splice site classification (one per species)
        self.splice_classifiers = nn.ModuleDict({
            name: nn.Conv1d(embed_dim, num_classes, kernel_size=1)
            for name in self.species_names
        })
        
        # Output heads for usage prediction (SSE) (one per species)
        self.usage_predictors = nn.ModuleDict({
            name: nn.Conv1d(embed_dim, n_conditions, kernel_size=1)
            for name in self.species_names
        })
        
        # Hybrid loss: add classification head for SSE (one per species)
        if usage_loss_type == 'hybrid':
            self.usage_classifiers = nn.ModuleDict({
                name: nn.Conv1d(embed_dim, n_conditions * 3, kernel_size=1)
                for name in self.species_names
            })
        
    def forward(self, sequences, species_ids=None, return_features: bool = False):
        """
        Forward pass through the model: Encoder → Transformer → Central Extraction → Output Heads.
        
        Args:
            sequences: One-hot encoded DNA sequences of shape (batch_size, seq_len, 4)
                      Format: [context_len | central_region | context_len]
            species_ids: Species IDs (batch,) - integer species identifiers
                        Used to select appropriate species-specific heads
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary with predictions for central region:
                - 'splice_logits': (batch_size, central_len, num_classes)
                - 'usage_predictions': (batch_size, central_len, n_conditions)
                - 'usage_class_logits': (batch_size, central_len, n_conditions, 3) [if usage_loss_type='hybrid']
            If return_features=True, also includes intermediate features
        """
        # Encoder: Conv blocks + multi-scale fusion
        encoder_output = self.encoder(sequences, return_features=return_features)
        full_features = encoder_output['features']  # (batch_size, seq_len, embed_dim)
        
        # Transformer: Multi-head self-attention on full sequence
        transformer_output = self.transformer(full_features)  # (batch_size, seq_len, embed_dim)
        
        # Extract central region after transformer (remove context from both ends)
        if self.context_len > 0:
            central_features = transformer_output[:, self.context_len:-self.context_len, :]  # (batch_size, central_len, embed_dim)
        else:
            central_features = transformer_output
        
        # Get species ID for output head selection
        if species_ids is None:
            species_id = 0
        else:
            if isinstance(species_ids, torch.Tensor):
                species_id = species_ids[0].item()
            else:
                species_id = species_ids[0] if hasattr(species_ids, '__getitem__') else species_ids
        
        species_name = self.species_names[species_id]
        
        # Transpose for Conv1d: (batch_size, embed_dim, central_len)
        transformer_conv = transformer_output.transpose(1, 2)
        
        # Apply output heads
        splice_logits = self.splice_classifiers[species_name](transformer_conv)
        usage_predictions = self.usage_predictors[species_name](transformer_conv)
        
        # Transpose back: (batch_size, central_len, num_classes/n_conditions)
        splice_logits = splice_logits.transpose(1, 2)
        usage_predictions = usage_predictions.transpose(1, 2)
        
        output = {
            'splice_logits': splice_logits,
            'usage_predictions': usage_predictions
        }
        
        # Hybrid loss: classification head for SSE
        if self.usage_loss_type == 'hybrid':
            usage_class_logits = self.usage_classifiers[species_name](transformer_conv)
            usage_class_logits = usage_class_logits.transpose(1, 2)
            batch_size, central_len, _ = usage_class_logits.shape
            usage_class_logits = usage_class_logits.view(batch_size, central_len, self.n_conditions, 3)
            output['usage_class_logits'] = usage_class_logits
        
        if return_features:
            output['encoder_features'] = encoder_output.get('encoder_features')
            output['central_features'] = central_features
            output['transformer_features'] = transformer_output
            output['skip_features'] = encoder_output.get('skip_features')
            
        return output
    
    def add_transform(self, transform):
        """Add a prediction transform function"""
        self.prediction_transform = transform
    
    def set_output_type(self, output_type: str = 'splice'):
        """Set which output to use for predictions"""
        if output_type not in ['splice', 'usage']:
            raise ValueError(f"Unknown output_type: {output_type}")
        self.output_type = output_type
    
    def predict(self, sequences, species_ids=None, batch_size: int = 32):
        """
        Predict splice sites and usage statistics for central region with batching.
        
        Args:
            sequences: One-hot encoded DNA sequences of shape (batch_size, seq_len, 4)
                      or (seq_len, 4) for single sequence
                      Can be numpy array, memmap, or torch tensor
                      Format: [context_len | central_region | context_len]
            species_ids: Species IDs (batch,) or single integer
            batch_size: Batch size for processing (default: 32)
            
        Returns:
            Dictionary with numpy arrays for central region
        """
        import numpy as np
        
        self.eval()
        
        # Convert to tensor if needed
        if isinstance(sequences, np.ndarray):
            sequences_is_memmap = isinstance(sequences, np.memmap) or not sequences.flags.writeable
            sequences_array = sequences
        else:
            sequences_is_memmap = False
        
        # Handle single sequence input
        single_input = False
        if not sequences_is_memmap and sequences.ndim == 2:
            sequences = sequences.unsqueeze(0)
            single_input = True
        elif sequences_is_memmap and sequences.ndim == 2:
            sequences_array = sequences[np.newaxis, ...]
            single_input = True
        
        # Handle species_ids
        if species_ids is not None:
            if isinstance(species_ids, (int, np.integer)):
                # Single species ID - broadcast to all sequences
                if sequences_is_memmap:
                    num_sequences = sequences_array.shape[0]
                else:
                    num_sequences = sequences.shape[0]
                species_ids = np.full(num_sequences, species_ids, dtype=np.int32)
            elif isinstance(species_ids, np.ndarray):
                pass  # Already an array
            else:
                species_ids = np.array(species_ids, dtype=np.int32)
        
        # Get number of sequences
        if sequences_is_memmap:
            num_sequences = sequences_array.shape[0]
        else:
            num_sequences = sequences.shape[0]
        
        device = next(self.parameters()).device
        
        # Initialize lists to collect results
        all_splice_logits = []
        all_splice_probs = []
        all_splice_predictions = []
        all_usage_predictions = []
        
        with torch.no_grad():
            # Process in batches
            for i in range(0, num_sequences, batch_size):
                batch_end = min(i + batch_size, num_sequences)
                
                if sequences_is_memmap:
                    batch_sequences = np.array(sequences_array[i:batch_end], dtype=np.float32)
                    batch_sequences = torch.from_numpy(batch_sequences)
                else:
                    batch_sequences = sequences[i:batch_end]
                
                # Get batch species IDs
                if species_ids is not None:
                    batch_species = torch.from_numpy(species_ids[i:batch_end])
                else:
                    batch_species = None
                
                # Transfer to device
                batch_sequences = batch_sequences.to(device)
                if batch_species is not None:
                    batch_species = batch_species.to(device)
                
                # Forward pass
                output = self.forward(batch_sequences, species_ids=batch_species)
                
                # Process predictions
                splice_logits = output['splice_logits']
                splice_probs = F.softmax(splice_logits, dim=-1)
                splice_predictions = splice_logits.argmax(dim=-1)
                usage_predictions = output['usage_predictions']
                
                # Move to CPU and convert to numpy
                all_splice_logits.append(splice_logits.cpu().numpy())
                all_splice_probs.append(splice_probs.cpu().numpy())
                all_splice_predictions.append(splice_predictions.cpu().numpy())
                all_usage_predictions.append(usage_predictions.cpu().numpy())
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Concatenate all batches
        splice_logits = np.concatenate(all_splice_logits, axis=0)
        splice_probs = np.concatenate(all_splice_probs, axis=0)
        splice_predictions = np.concatenate(all_splice_predictions, axis=0)
        usage_predictions = np.concatenate(all_usage_predictions, axis=0)
        
        # Remove batch dimension if single input
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
