import torch
import pandas as pd
from .model import SplicevoModel
from ..training import SpliceTrainer, SpliceDataset
from torch.utils.data import DataLoader


class Splicevo:
    """
    Main class to handle training, testing, predicting and calculation of contribution scores.
    """

    def __init__(
            self,
            data: pd.DataFrame | None = None,
            model: torch.nn.Module | None = None,
            config: dict | None = None
        ):
        """Initialize Splicevo with data, model, and configuration.""" 
        self.data = data
        self.config = config if config else {}
        
        # Initialize model
        if model is None:
            self.model = SplicevoModel(
                embed_dim=self.config.get('embed_dim', 256),
                num_resblocks=self.config.get('num_resblocks', 4),
                dilation_strategy=self.config.get('dilation_strategy', 'exponential'),
                num_classes=self.config.get('num_classes', 3),
                n_conditions=self.config.get('n_conditions', 5),
                dropout=self.config.get('dropout', 0.1),
                usage_loss_type=self.config.get('usage_loss_type', 'mse')
            )
        else:
            self.model = model
        
        self.trainer = None
    
    def create_trainer(
        self,
        train_data: tuple,
        val_data: tuple | None = None,
        batch_size: int = 32,
        n_species: int = 1,
        balance_species: bool = False,
        **trainer_kwargs
    ):
        """
        Create a trainer for the model.
        
        Args:
            train_data: Tuple of (sequences, labels, usage_sse, species_ids)
            val_data: Optional validation data in same format
            batch_size: Batch size for training
            n_species: Number of species in the data (1 for single-species, >1 for multi-species)
            balance_species: Whether to balance species representation in batches
                           Only applies when n_species > 1
            **trainer_kwargs: Additional arguments for SpliceTrainer
        """
        # Create datasets
        train_dataset = SpliceDataset(*train_data)
        
        # For multi-species training, use species-specific batch sampler
        if n_species > 1:
            from ..training.samplers import SpeciesBatchSampler
            
            train_sampler = SpeciesBatchSampler(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                balance_species=balance_species,
                drop_last=True,
                seed=42
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                num_workers=4,
                pin_memory=True
            )
        else:
            # Single-species: use regular DataLoader
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
        
        val_loader = None
        if val_data is not None:
            val_dataset = SpliceDataset(*val_data)
            
            # Use same sampling strategy for validation
            if n_species > 1:
                from ..training.samplers import SpeciesBatchSampler
                
                val_sampler = SpeciesBatchSampler(
                    dataset=val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    balance_species=balance_species,
                    drop_last=False,
                    seed=42
                )
                
                val_loader = DataLoader(
                    val_dataset,
                    batch_sampler=val_sampler,
                    num_workers=4,
                    pin_memory=True
                )
            else:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True
                )
        
        # Create trainer
        self.trainer = SpliceTrainer(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            **trainer_kwargs
        )
        
        return self.trainer
    
    def train(self, n_epochs: int, **kwargs):
        """Train the model."""
        if self.trainer is None:
            raise ValueError("Trainer not created. Call create_trainer() first.")
        
        self.trainer.train(n_epochs, **kwargs)
    
    def predict(self, sequences, **kwargs):
        """Predict splice sites and usage in sequences."""
        return self.model.predict(sequences, **kwargs)
    
    def forward(self, sequences, **kwargs):
        """Forward pass through the model."""
        return self.model(sequences, **kwargs)

