import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np

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
                 alternate: Optional[int] = 4,
                 num_classes: int = 3,
                 n_conditions: int = 5,
                 add_output_heads: bool = True,
                 context_len: int = 4500,
                 dropout: float = 0.0,
                 usage_loss_type: str = 'weighted_mse',
                 n_species: int = 1,
                 species_names: Optional[list] = None  # CHANGED: add species names
        ):
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
            usage_loss_type: Type of usage loss ('mse', 'weighted_mse', 'hybrid')
            n_species: Number of species (creates separate heads per species)
            species_names: Optional list of species names for tracking
        """
        super().__init__()
        
        self.n_conditions = n_conditions
        self.add_output_heads = add_output_heads
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
        
        # Optional output heads - CREATE SEPARATE HEADS PER SPECIES
        if add_output_heads:
            # Splice site classification heads (one per species)
            self.splice_classifiers = nn.ModuleDict({
                name: nn.Conv1d(embed_dim, num_classes, kernel_size=1)
                for name in self.species_names
            })
            
            # Usage prediction heads - SSE only (one per species)
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
            If add_output_heads=True:
                Dictionary with predictions only for the central region:
                    - 'splice_logits': (batch_size, central_len, num_classes)
                    - 'usage_predictions': (batch_size, central_len, n_conditions)
                    - 'usage_class_logits': (batch_size, central_len, n_conditions, 3) [if usage_loss_type='hybrid']
                If return_features=True, also includes:
                    - 'encoder_features': Full features (batch_size, seq_len, embed_dim)
                    - 'central_features': Central region features (batch_size, central_len, embed_dim)
                    - 'skip_features': Fused skip features (batch_size, seq_len, embed_dim)
            If add_output_heads=False:
                Central region features of shape (batch_size, central_len, embed_dim)
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
        
        # Extract central region (remove context from both ends)
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
        
        # SPECIES-SPECIFIC HEADS: Select appropriate head based on species_ids
        # During training, all sequences in batch should be from same species
        # During inference, can process mixed batches by grouping
        
        if species_ids is None:
            # Default to first species if not specified
            species_id = 0
        else:
            # Get species ID from batch (should be same for all in training batch)
            if isinstance(species_ids, torch.Tensor):
                species_id = species_ids[0].item()
            else:
                species_id = species_ids[0] if hasattr(species_ids, '__getitem__') else species_ids
        
        species_name = self.species_names[species_id]
        
        # Apply species-specific output heads
        splice_logits = self.splice_classifiers[species_name](central_skip_conv)
        usage_predictions = self.usage_predictors[species_name](central_skip_conv)
        
        # Transpose back: (batch, central_len, num_classes/n_conditions)
        splice_logits = splice_logits.transpose(1, 2)
        usage_predictions = usage_predictions.transpose(1, 2)
        
        # Apply sigmoid activation to SSE
        usage_predictions = torch.sigmoid(usage_predictions)
        
        output = {
            'splice_logits': splice_logits,
            'usage_predictions': usage_predictions
        }
        
        # Output 3: Classify SSE values (for hybrid loss)
        if self.usage_loss_type == 'hybrid':
            usage_class_logits = self.usage_classifiers[species_name](central_skip_conv)
            usage_class_logits = usage_class_logits.transpose(1, 2)
            
            # Reshape: (batch, central_len, n_conditions, 3)
            batch_size, central_len, _ = usage_class_logits.shape
            usage_class_logits = usage_class_logits.view(batch_size, central_len, self.n_conditions, 3)
            output['usage_class_logits'] = usage_class_logits
        
        if return_features:
            output['encoder_features'] = encoder_features
            output['central_features'] = central_features
            output['skip_features'] = fused_skip
            
        return output


class SplicevoModel(nn.Module):
    """Simple model with encoder and dual decoders for splice site and usage prediction."""
    
    def __init__(self, 
                 embed_dim: int = 256, 
                 num_resblocks: int = 4,
                 dilation_strategy: str = 'exponential',
                 alternate: Optional[int] = 4,
                 num_classes: int = 3,
                 n_conditions: int = 5,
                 context_len: int = 4500,
                 dropout: float = 0.3,
                 usage_loss_type: str = 'weighted_mse',
                 n_species: int = 1,
                 species_names: Optional[list] = None
        ):
        """
        Initialize model with encoder for splice site and SSE prediction.
        
        Args:
            n_conditions: Number of conditions (tissues/timepoints) for SSE prediction
            usage_loss_type: Type of usage loss ('mse', 'weighted_mse', 'hybrid')
            n_species: Number of species (creates separate output heads per species)
            species_names: Optional list of species names
        """
        super().__init__()
        
        self.context_len = context_len
        self.usage_loss_type = usage_loss_type
        self.n_species = n_species
        self.species_names = species_names or [f"species_{i}" for i in range(n_species)]
        self.output_type = 'splice'  # For gReLU compatibility
        self.prediction_transform = None  # For gReLU compatibility
        
        # Encoder module with output heads
        self.encoder = EncoderModule(
            embed_dim=embed_dim,
            num_resblocks=num_resblocks,
            dilation_strategy=dilation_strategy,
            alternate=alternate,
            num_classes=num_classes,
            n_conditions=n_conditions,
            add_output_heads=True,
            context_len=context_len,
            dropout=dropout,
            usage_loss_type=usage_loss_type,
            n_species=n_species,
            species_names=species_names
        )
        
    def forward(self, sequences, species_ids=None, return_features: bool = False):
        """
        Forward pass through the model.
        
        Args:
            sequences: One-hot encoded DNA sequences of shape (batch_size, seq_len, 4)
                      Format: [context_len | central_region | context_len]
            species_ids: Species IDs (batch,) - integer species identifiers
                        Used to select appropriate species-specific heads
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary with predictions for central region only
        """
        output = self.encoder(sequences, species_ids=species_ids, return_features=return_features)
        
        # Always return full dictionary - Captum will handle gradient computation
        return output
    
    def add_transform(self, transform):
        """Add a prediction transform function (for GRELU compatibility)."""
        self.prediction_transform = transform
    
    def set_output_type(self, output_type: str = 'splice'):
        """Set which output to use for predictions (for GRELU compatibility)."""
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
