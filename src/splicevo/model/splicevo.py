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
        **trainer_kwargs
    ):
        """
        Create a trainer for the model.
        
        Args:
            train_data: Tuple of (sequences, labels, usage_sse)
            val_data: Optional validation data in same format
            batch_size: Batch size for training
            **trainer_kwargs: Additional arguments for SpliceTrainer
        """
        # Create datasets
        train_dataset = SpliceDataset(*train_data)
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

