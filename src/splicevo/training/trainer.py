"""Training module for splice site prediction models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
import time

from .losses import WeightedMSELoss, HybridUsageLoss


class SpliceTrainer:
    """
    Trainer class for splice site prediction with multi-task learning.
    
    Handles training, validation, checkpointing, and logging for models
    with dual decoders (splice site classification + usage prediction).
    Supports memory-mapped datasets for efficient large-scale training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        splice_weight: float = 1.0,
        usage_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        checkpoint_dir: Optional[str] = None,
        use_tensorboard: bool = True,
        pin_memory: bool = True,
        non_blocking: bool = True,
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        usage_loss_type: str = 'weighted_mse',
        weighted_mse_extreme_low: float = 0.05,
        weighted_mse_extreme_high: float = 0.95,
        weighted_mse_extreme_weight: float = 10.0,
        hybrid_extreme_low: float = 0.05,
        hybrid_extreme_high: float = 0.95,
        hybrid_class_weight: float = 1.0,
        hybrid_reg_weight: float = 1.0
    ):
        """
        Initialize trainer.
        
        Args:
            model: The model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            splice_weight: Weight for splice classification loss contribution to total loss
            usage_weight: Weight for usage prediction loss contribution to total loss
            class_weights: Weights for each splice site class (for imbalanced data)
            checkpoint_dir: Directory to save checkpoints
            pin_memory: Pin memory for faster CPU-GPU transfer (for memmap data)
            non_blocking: Use non-blocking transfers for better performance
            use_amp: Use automatic mixed precision training (reduces memory usage)
            gradient_accumulation_steps: Number of steps to accumulate gradients before update
            usage_loss_type: Type of usage loss ('mse', 'weighted_mse', 'hybrid')
            weighted_mse_extreme_threshold: Threshold for extreme values (for weighted_mse)
            weighted_mse_extreme_weight: Weight for extreme values (for weighted_mse)
            hybrid_extreme_low: Low threshold for hybrid loss classification
            hybrid_extreme_high: High threshold for hybrid loss classification
            hybrid_class_weight: Weight for classification component in hybrid loss
            hybrid_reg_weight: Weight for regression component in hybrid loss
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_tensorboard = use_tensorboard and (checkpoint_dir is not None)
        self.pin_memory = pin_memory
        self.non_blocking = non_blocking
        self.use_amp = use_amp and (device == 'cuda')
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.usage_loss_type = usage_loss_type
        
        # Mixed precision training
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # Loss functions
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.splice_criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Usage loss: weighted MSE, hybrid, or regular MSE
        if usage_loss_type == 'weighted_mse':
            self.usage_criterion = WeightedMSELoss(
                extreme_low_threshold=weighted_mse_extreme_low,
                extreme_high_threshold=weighted_mse_extreme_high,
                extreme_weight=weighted_mse_extreme_weight,
                reduction='mean'
            )
        elif usage_loss_type == 'hybrid':
            self.usage_criterion = HybridUsageLoss(
                extreme_low_threshold=hybrid_extreme_low,
                extreme_high_threshold=hybrid_extreme_high,
                class_weight=hybrid_class_weight,
                reg_weight=hybrid_reg_weight,
                reduction='mean'
            )
        else:  # 'mse' or default
            self.usage_criterion = nn.MSELoss()
        
        # Loss weights for multi-task learning
        self.splice_weight = splice_weight
        self.usage_weight = usage_weight
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Adjust scheduler for gradient accumulation
        total_steps = len(train_loader) * 100 // gradient_accumulation_steps
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps
        )
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard logging
        self.writer = None
        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.checkpoint_dir / 'tensorboard')
            
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'train_splice_loss': [],
            'train_usage_loss': [],
            'val_loss': [],
            'val_splice_loss': [],
            'val_usage_loss': [],
            'learning_rate': []
        }
        
        # Loss tracking for different SSE ranges (weighted_mse)
        self.sse_loss_tracking = {
            'train': {},  # epoch -> {'zeros': [], 'ones': [], 'middle': [], 'targets': [], 'predictions': []}
            'val': {}
        }
        
        # Loss tracking for splice site classes
        self.splice_class_tracking = {
            'train': {},  # epoch -> {'class_0': loss, 'class_1': loss, 'class_2': loss}
            'val': {}
        }
        
        # Loss tracking for hybrid usage loss components
        self.hybrid_loss_tracking = {
            'train': {},  # epoch -> {'regression': loss, 'classification': loss, 'class_zeros': loss, 'class_ones': loss, 'class_middle': loss}
            'val': {}
        }
        
        # Time tracking
        self.epoch_start_time = None
        self.first_epoch_duration = None
        self.training_start_time = None
    
    def _track_sse_losses(
        self, 
        usage_predictions: torch.Tensor,
        usage_targets: torch.Tensor,
        splice_mask: torch.Tensor,
        usage_class_logits: Optional[torch.Tensor] = None,
        is_training: bool = True
    ) -> Dict:
        """
        Track losses for different SSE value ranges.
        
        Works with all loss types: MSE, Weighted MSE, Hybrid, BCE.
        
        Args:
            usage_predictions: Predicted SSE values (batch, central_len, n_conditions)
            usage_targets: Target SSE values (batch, central_len, n_conditions)
            splice_mask: Mask for valid splice sites (batch, central_len)
            usage_class_logits: Classification logits for hybrid loss (batch, central_len, n_conditions, 3)
            is_training: Whether this is training or validation
            
        Returns:
            Dictionary with losses and predictions/targets for different ranges
        """
        if splice_mask.sum() == 0:
            return None
        
        # Flatten and mask
        mask_flat = splice_mask.reshape(-1)
        usage_targets_flat = usage_targets.reshape(-1, usage_targets.shape[-1])
        usage_predictions_flat = usage_predictions.reshape(-1, usage_predictions.shape[-1])
        
        masked_targets = usage_targets_flat[mask_flat]
        masked_preds = usage_predictions_flat[mask_flat]
        
        # Compute per-position losses based on loss type
        if self.usage_loss_type == 'hybrid':
            # Hybrid loss: need both regression and classification
            if usage_class_logits is None:
                # Fallback to MSE if class logits not provided
                per_pos_loss = (masked_preds - masked_targets) ** 2
            else:
                usage_class_flat = usage_class_logits.reshape(-1, usage_class_logits.shape[-2], usage_class_logits.shape[-1])
                masked_class_logits = usage_class_flat[mask_flat]
                
                # Compute hybrid loss per position
                original_reduction = self.usage_criterion.reduction
                self.usage_criterion.reduction = 'none'
                per_pos_loss = self.usage_criterion(masked_preds, masked_class_logits, masked_targets)
                self.usage_criterion.reduction = original_reduction
        
        elif self.usage_loss_type == 'weighted_mse':
            # Weighted MSE
            original_reduction = self.usage_criterion.reduction
            self.usage_criterion.reduction = 'none'
            per_pos_loss = self.usage_criterion(masked_preds, masked_targets)
            self.usage_criterion.reduction = original_reduction
        
        elif self.usage_loss_type == 'bce':
            # Binary cross-entropy (for binary classification)
            # Assumes targets are in [0, 1] and predictions are logits or probabilities
            if hasattr(self.usage_criterion, 'reduction'):
                original_reduction = self.usage_criterion.reduction
                self.usage_criterion.reduction = 'none'
                per_pos_loss = self.usage_criterion(masked_preds, masked_targets)
                self.usage_criterion.reduction = original_reduction
            else:
                # Manual BCE calculation if needed
                per_pos_loss = -(masked_targets * torch.log(masked_preds + 1e-8) + 
                                (1 - masked_targets) * torch.log(1 - masked_preds + 1e-8))
        
        else:  # 'mse' or default
            # Standard MSE
            per_pos_loss = (masked_preds - masked_targets) ** 2
        
        # Identify different SSE ranges
        is_zero = (masked_targets < 0.05)
        is_one = (masked_targets > 0.95)
        is_middle = (masked_targets >= 0.05) & (masked_targets <= 0.95)
        
        # Extract losses
        zero_losses = per_pos_loss[is_zero]
        one_losses = per_pos_loss[is_one]
        middle_losses = per_pos_loss[is_middle]
        
        return {
            'zeros': zero_losses.detach().cpu().numpy() if zero_losses.numel() > 0 else np.array([]),
            'ones': one_losses.detach().cpu().numpy() if one_losses.numel() > 0 else np.array([]),
            'middle': middle_losses.detach().cpu().numpy() if middle_losses.numel() > 0 else np.array([]),
            'n_zeros': is_zero.sum().item(),
            'n_ones': is_one.sum().item(),
            'n_middle': is_middle.sum().item(),
            'targets': masked_targets.detach().cpu().numpy(),
            'predictions': masked_preds.detach().cpu().numpy()
        }
    
    def _track_splice_class_losses(
        self,
        splice_logits: torch.Tensor,
        splice_labels: torch.Tensor,
        is_training: bool = True
    ) -> Dict:
        """
        Track losses for different splice site classes.
        
        Args:
            splice_logits: Predicted logits (batch, positions, n_classes)
            splice_labels: Target labels (batch, positions)
            is_training: Whether this is training or validation
            
        Returns:
            Dictionary with losses per class
        """
        result = {}
        
        # Flatten
        logits_flat = splice_logits.reshape(-1, splice_logits.size(-1))
        labels_flat = splice_labels.reshape(-1)
        
        # Compute loss per class
        for class_idx in range(splice_logits.size(-1)):
            mask = (labels_flat == class_idx)
            if mask.sum() > 0:
                # Use CrossEntropyLoss without reduction
                loss_fn = nn.CrossEntropyLoss(reduction='none')
                per_sample_loss = loss_fn(logits_flat, labels_flat)
                class_loss = per_sample_loss[mask].mean()
                result[f'class_{class_idx}'] = class_loss.item()
            else:
                result[f'class_{class_idx}'] = 0.0
        
        return result
    
    def _track_hybrid_class_losses(
        self,
        usage_predictions: torch.Tensor,
        usage_class_logits: torch.Tensor,
        usage_targets: torch.Tensor,
        splice_mask: torch.Tensor,
        is_training: bool = True
    ) -> Dict:
        """
        Track classification losses for different SSE ranges in hybrid loss.
        
        Args:
            usage_predictions: Predicted SSE values (batch, central_len, n_conditions)
            usage_class_logits: Classification logits (batch, central_len, n_conditions, 3)
            usage_targets: Target SSE values (batch, central_len, n_conditions)
            splice_mask: Mask for valid splice sites (batch, central_len)
            is_training: Whether this is training or validation
            
        Returns:
            Dictionary with classification losses for different ranges
        """
        if splice_mask.sum() == 0 or usage_class_logits is None:
            return {'zeros': 0.0, 'ones': 0.0, 'middle': 0.0}
        
        # Flatten and mask
        mask_flat = splice_mask.reshape(-1)
        usage_targets_flat = usage_targets.reshape(-1, usage_targets.shape[-1])
        usage_class_flat = usage_class_logits.reshape(-1, usage_class_logits.shape[-2], usage_class_logits.shape[-1])
        
        masked_targets = usage_targets_flat[mask_flat]
        masked_class_logits = usage_class_flat[mask_flat]
        
        # Identify different SSE ranges
        is_zero = (masked_targets < self.usage_criterion.extreme_low_threshold)
        is_one = (masked_targets > self.usage_criterion.extreme_high_threshold)
        is_middle = (masked_targets >= self.usage_criterion.extreme_low_threshold) & (masked_targets <= self.usage_criterion.extreme_high_threshold)
        
        # Get true class labels based on thresholds
        true_classes = torch.zeros_like(masked_targets, dtype=torch.long)
        true_classes[is_zero] = 0
        true_classes[is_middle] = 1
        true_classes[is_one] = 2
        
        # Compute classification loss per range
        result = {}
        ce_loss = nn.CrossEntropyLoss(reduction='none')
        
        # Reshape for CrossEntropyLoss: (N*n_conditions, 3) and (N*n_conditions)
        n_samples, n_conditions, n_classes = masked_class_logits.shape
        logits_2d = masked_class_logits.reshape(-1, n_classes)  # (N*n_conditions, 3)
        labels_1d = true_classes.reshape(-1)  # (N*n_conditions)
        
        # Compute per-sample classification loss
        per_sample_loss = ce_loss(logits_2d, labels_1d)  # (N*n_conditions)
        per_sample_loss = per_sample_loss.reshape(n_samples, n_conditions)  # (N, n_conditions)
        
        # Average over conditions, then separate by range
        per_position_loss = per_sample_loss.mean(dim=1)  # (N)
        
        if is_zero.any(dim=1).any():
            zero_mask = is_zero.any(dim=1)
            result['zeros'] = per_position_loss[zero_mask].mean().item()
        else:
            result['zeros'] = 0.0
        
        if is_one.any(dim=1).any():
            one_mask = is_one.any(dim=1)
            result['ones'] = per_position_loss[one_mask].mean().item()
        else:
            result['ones'] = 0.0
        
        if is_middle.any(dim=1).any():
            middle_mask = is_middle.any(dim=1)
            result['middle'] = per_position_loss[middle_mask].mean().item()
        else:
            result['middle'] = 0.0
        
        return result
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_splice_loss = 0
        total_usage_loss = 0
        n_batches = 0
        
        # Track SSE losses for this epoch
        epoch_sse_tracking = {
            'zeros': [],
            'ones': [],
            'middle': [],
            'n_zeros': 0,
            'n_ones': 0,
            'n_middle': 0,
            'targets': [],
            'predictions': []
        }
        
        # Track splice class losses
        epoch_splice_class = {'class_0': [], 'class_1': [], 'class_2': []}
        
        # Track hybrid loss components
        epoch_hybrid_tracking = {'regression': [], 'classification': [], 'class_zeros': [], 'class_ones': [], 'class_middle': []}
        
        # Reset gradients at the start
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            sequences = batch['sequences'].to(self.device, non_blocking=self.non_blocking)
            splice_labels = batch['splice_labels'].to(self.device, non_blocking=self.non_blocking)
            usage_targets = batch['usage_targets'].to(self.device, non_blocking=self.non_blocking)
            species_ids = batch['species_id'].to(self.device, non_blocking=self.non_blocking)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                output = self.model(sequences, species_ids=species_ids)
                
                # Compute splice classification loss
                splice_logits = output['splice_logits']
                splice_loss = self.splice_criterion(
                    splice_logits.reshape(-1, splice_logits.size(-1)),
                    splice_labels.reshape(-1)
                )
                
                # Track splice class losses
                splice_class_losses = self._track_splice_class_losses(
                    splice_logits, splice_labels, is_training=True
                )
                for class_name, loss_val in splice_class_losses.items():
                    epoch_splice_class[class_name].append(loss_val)
                
                # Mask for valid splice positions
                splice_mask = (splice_labels > 0)  # shape: [batch, positions]
                usage_targets = torch.nan_to_num(usage_targets, nan=0.0)
                
                # Compute usage loss based on loss type
                if self.usage_loss_type == 'hybrid':
                    # Hybrid loss expects both regression and classification outputs
                    usage_predictions = output['usage_predictions']  # (batch, central_len, n_conditions)
                    usage_class_logits = output.get('usage_class_logits', None)  # (batch, central_len, n_conditions, 3)
                    
                    if usage_class_logits is None:
                        raise ValueError("Model must output 'usage_class_logits' for hybrid loss")
                    
                    if splice_mask.sum() > 0:
                        mask_flat = splice_mask.reshape(-1)
                        # Flatten first two dims (batch, central_len), keep n_conditions
                        usage_preds_flat = usage_predictions.reshape(-1, usage_predictions.shape[-1])
                        usage_class_flat = usage_class_logits.reshape(-1, usage_class_logits.shape[-2], usage_class_logits.shape[-1])
                        usage_targets_flat = usage_targets.reshape(-1, usage_targets.shape[-1])
                        
                        # Get component losses
                        original_reduction = self.usage_criterion.reduction
                        self.usage_criterion.reduction = 'none'
                        reg_loss, class_loss = self.usage_criterion.get_component_losses(
                            usage_preds_flat[mask_flat],
                            usage_class_flat[mask_flat],
                            usage_targets_flat[mask_flat]
                        )
                        self.usage_criterion.reduction = original_reduction
                        
                        # Track components
                        epoch_hybrid_tracking['regression'].append(reg_loss.mean().item())
                        epoch_hybrid_tracking['classification'].append(class_loss.mean().item())
                        
                        # Track classification loss by SSE range
                        hybrid_class_losses = self._track_hybrid_class_losses(
                            usage_predictions, usage_class_logits, usage_targets, splice_mask, is_training=True
                        )
                        epoch_hybrid_tracking['class_zeros'].append(hybrid_class_losses['zeros'])
                        epoch_hybrid_tracking['class_ones'].append(hybrid_class_losses['ones'])
                        epoch_hybrid_tracking['class_middle'].append(hybrid_class_losses['middle'])
                        
                        # Combined loss
                        usage_loss = self.usage_criterion(
                            usage_preds_flat[mask_flat],
                            usage_class_flat[mask_flat],
                            usage_targets_flat[mask_flat]
                        )
                        
                        # Track SSE losses (pass class logits for hybrid)
                        sse_tracking = self._track_sse_losses(
                            usage_predictions, usage_targets, splice_mask, 
                            usage_class_logits=usage_class_logits,
                            is_training=True
                        )
                        if sse_tracking is not None:
                            epoch_sse_tracking['zeros'].extend(sse_tracking['zeros'].tolist())
                            epoch_sse_tracking['ones'].extend(sse_tracking['ones'].tolist())
                            epoch_sse_tracking['middle'].extend(sse_tracking['middle'].tolist())
                            epoch_sse_tracking['n_zeros'] += sse_tracking['n_zeros']
                            epoch_sse_tracking['n_ones'] += sse_tracking['n_ones']
                            epoch_sse_tracking['n_middle'] += sse_tracking['n_middle']
                            epoch_sse_tracking['targets'].append(sse_tracking['targets'])
                            epoch_sse_tracking['predictions'].append(sse_tracking['predictions'])
                    else:
                        usage_loss = torch.tensor(0.0, device=self.device)
                elif self.usage_loss_type == 'weighted_mse':
                    # Weighted MSE for SSE
                    usage_predictions = output['usage_predictions']  # (batch, central_len, n_conditions)
                    if splice_mask.sum() > 0:
                        mask_flat = splice_mask.reshape(-1)
                        usage_targets_flat = usage_targets.reshape(-1, usage_targets.shape[-1])
                        usage_predictions_flat = usage_predictions.reshape(-1, usage_predictions.shape[-1])
                        usage_loss = self.usage_criterion(
                            usage_predictions_flat[mask_flat],
                            usage_targets_flat[mask_flat]
                        )
                        
                        # Track SSE losses (no class logits for weighted MSE)
                        sse_tracking = self._track_sse_losses(
                            usage_predictions, usage_targets, splice_mask, 
                            usage_class_logits=None,
                            is_training=True
                        )
                        if sse_tracking is not None:
                            epoch_sse_tracking['zeros'].extend(sse_tracking['zeros'].tolist())
                            epoch_sse_tracking['ones'].extend(sse_tracking['ones'].tolist())
                            epoch_sse_tracking['middle'].extend(sse_tracking['middle'].tolist())
                            epoch_sse_tracking['n_zeros'] += sse_tracking['n_zeros']
                            epoch_sse_tracking['n_ones'] += sse_tracking['n_ones']
                            epoch_sse_tracking['n_middle'] += sse_tracking['n_middle']
                            epoch_sse_tracking['targets'].append(sse_tracking['targets'])
                            epoch_sse_tracking['predictions'].append(sse_tracking['predictions'])
                    else:
                        usage_loss = torch.tensor(0.0, device=self.device)
                else:
                    # Standard MSE or BCE
                    usage_predictions = output['usage_predictions']  # (batch, central_len, n_conditions)
                    if splice_mask.sum() > 0:
                        mask_flat = splice_mask.reshape(-1)
                        usage_targets_flat = usage_targets.reshape(-1, usage_targets.shape[-1])
                        usage_predictions_flat = usage_predictions.reshape(-1, usage_predictions.shape[-1])
                        usage_loss = self.usage_criterion(
                            usage_predictions_flat[mask_flat],
                            usage_targets_flat[mask_flat]
                        )
                    else:
                        usage_loss = torch.tensor(0.0, device=self.device)
            
            # Combined loss (scale by accumulation steps)
            loss = (
                self.splice_weight * splice_loss +
                self.usage_weight * usage_loss
            ) / self.gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    # Unscale gradients and clip
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # Log to TensorBoard (per update step, not per batch)
                if self.writer is not None:
                    # Multiply loss back for logging
                    actual_loss = loss.item() * self.gradient_accumulation_steps
                    self.writer.add_scalar('Train/Loss_Batch', actual_loss, self.global_step)
                    self.writer.add_scalar('Train/Splice_Loss_Batch', splice_loss.item(), self.global_step)
                    self.writer.add_scalar('Train/Usage_Loss_Batch', usage_loss.item(), self.global_step)
                    self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], self.global_step)
                    
                    # Log gradient norms every N steps
                    if self.global_step % 100 == 0:
                        total_norm = 0.0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5
                        self.writer.add_scalar('Train/Gradient_Norm', total_norm, self.global_step)
                
                self.global_step += 1
            
            # Accumulate losses (multiply back for accurate tracking)
            total_loss += loss.item() * self.gradient_accumulation_steps
            total_splice_loss += splice_loss.item()
            total_usage_loss += usage_loss.item()
            n_batches += 1
            
            # Clear CUDA cache periodically
            if self.device == 'cuda' and batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        # Store splice class tracking for this epoch
        self.splice_class_tracking['train'][self.current_epoch + 1] = {
            'class_0': np.mean(epoch_splice_class['class_0']) if epoch_splice_class['class_0'] else 0.0,
            'class_1': np.mean(epoch_splice_class['class_1']) if epoch_splice_class['class_1'] else 0.0,
            'class_2': np.mean(epoch_splice_class['class_2']) if epoch_splice_class['class_2'] else 0.0,
        }
        
        # Store hybrid tracking for this epoch
        if self.usage_loss_type == 'hybrid' and epoch_hybrid_tracking['regression']:
            self.hybrid_loss_tracking['train'][self.current_epoch + 1] = {
                'regression': np.mean(epoch_hybrid_tracking['regression']),
                'classification': np.mean(epoch_hybrid_tracking['classification']),
                'class_zeros': np.mean([x for x in epoch_hybrid_tracking['class_zeros'] if x > 0]) if any(x > 0 for x in epoch_hybrid_tracking['class_zeros']) else 0.0,
                'class_ones': np.mean([x for x in epoch_hybrid_tracking['class_ones'] if x > 0]) if any(x > 0 for x in epoch_hybrid_tracking['class_ones']) else 0.0,
                'class_middle': np.mean([x for x in epoch_hybrid_tracking['class_middle'] if x > 0]) if any(x > 0 for x in epoch_hybrid_tracking['class_middle']) else 0.0
            }
        
        # Store SSE tracking for this epoch
        self.sse_loss_tracking['train'][self.current_epoch + 1] = {
            'zeros': np.array(epoch_sse_tracking['zeros']),
            'ones': np.array(epoch_sse_tracking['ones']),
            'middle': np.array(epoch_sse_tracking['middle']),
            'n_zeros': epoch_sse_tracking['n_zeros'],
            'n_ones': epoch_sse_tracking['n_ones'],
            'n_middle': epoch_sse_tracking['n_middle'],
            'targets': np.concatenate(epoch_sse_tracking['targets']) if epoch_sse_tracking['targets'] else np.array([]),
            'predictions': np.concatenate(epoch_sse_tracking['predictions']) if epoch_sse_tracking['predictions'] else np.array([])
        }
        
        # Log splice class losses to TensorBoard
        if self.writer is not None:
            epoch_num = self.current_epoch
            
            # Splice class losses
            class_tracking = self.splice_class_tracking['train'][self.current_epoch + 1]
            self.writer.add_scalar('Splice_Loss/Train_Class_0', class_tracking['class_0'], epoch_num)
            self.writer.add_scalar('Splice_Loss/Train_Class_1', class_tracking['class_1'], epoch_num)
            self.writer.add_scalar('Splice_Loss/Train_Class_2', class_tracking['class_2'], epoch_num)
            
            self.writer.add_scalars('Splice_Loss/Train_All_Classes', {
                'class_0': class_tracking['class_0'],
                'class_1': class_tracking['class_1'],
                'class_2': class_tracking['class_2']
            }, epoch_num)
            
            # Hybrid loss components
            if self.usage_loss_type == 'hybrid' and (self.current_epoch + 1) in self.hybrid_loss_tracking['train']:
                hybrid_tracking = self.hybrid_loss_tracking['train'][self.current_epoch + 1]
                self.writer.add_scalar('Hybrid_Loss/Train_Regression', hybrid_tracking['regression'], epoch_num)
                self.writer.add_scalar('Hybrid_Loss/Train_Classification', hybrid_tracking['classification'], epoch_num)
                
                self.writer.add_scalars('Hybrid_Loss/Train_Components', {
                    'regression': hybrid_tracking['regression'],
                    'classification': hybrid_tracking['classification']
                }, epoch_num)
                
                # Classification losses by SSE range
                self.writer.add_scalar('Hybrid_Classification/Train_Zeros', hybrid_tracking['class_zeros'], epoch_num)
                self.writer.add_scalar('Hybrid_Classification/Train_Ones', hybrid_tracking['class_ones'], epoch_num)
                self.writer.add_scalar('Hybrid_Classification/Train_Middle', hybrid_tracking['class_middle'], epoch_num)
                
                self.writer.add_scalars('Hybrid_Classification/Train_By_Range', {
                    'zeros': hybrid_tracking['class_zeros'],
                    'ones': hybrid_tracking['class_ones'],
                    'middle': hybrid_tracking['class_middle']
                }, epoch_num)
            
            # Weighted MSE SSE tracking
            if self.usage_loss_type == 'weighted_mse':
                if len(epoch_sse_tracking['zeros']) > 0:
                    self.writer.add_scalar('Usage_Loss/Train_Zeros', 
                                          np.mean(epoch_sse_tracking['zeros']), epoch_num)
                if len(epoch_sse_tracking['ones']) > 0:
                    self.writer.add_scalar('Usage_Loss/Train_Ones', 
                                          np.mean(epoch_sse_tracking['ones']), epoch_num)
                if len(epoch_sse_tracking['middle']) > 0:
                    self.writer.add_scalar('Usage_Loss/Train_Middle', 
                                          np.mean(epoch_sse_tracking['middle']), epoch_num)
                
                self.writer.add_scalars('Usage_Loss/Train_Combined', {
                    'zeros': np.mean(epoch_sse_tracking['zeros']) if len(epoch_sse_tracking['zeros']) > 0 else 0.0,
                    'ones': np.mean(epoch_sse_tracking['ones']) if len(epoch_sse_tracking['ones']) > 0 else 0.0,
                    'middle': np.mean(epoch_sse_tracking['middle']) if len(epoch_sse_tracking['middle']) > 0 else 0.0
                }, epoch_num)
            
            # Correlation
            if len(epoch_sse_tracking['targets']) > 0 and len(epoch_sse_tracking['predictions']) > 0:
                targets = np.concatenate(epoch_sse_tracking['targets']).flatten()
                preds = np.concatenate(epoch_sse_tracking['predictions']).flatten()
                correlation = np.corrcoef(targets, preds)[0, 1]
                self.writer.add_scalar('Usage_Metrics/Train_Correlation', correlation, epoch_num)
        
        return {
            'loss': total_loss / n_batches,
            'splice_loss': total_splice_loss / n_batches,
            'usage_loss': total_usage_loss / n_batches
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0
        total_splice_loss = 0
        total_usage_loss = 0
        n_batches = 0
        
        # Track SSE losses for this epoch
        epoch_sse_tracking = {
            'zeros': [],
            'ones': [],
            'middle': [],
            'n_zeros': 0,
            'n_ones': 0,
            'n_middle': 0,
            'targets': [],
            'predictions': []
        }
        
        # Track splice class losses
        epoch_splice_class = {'class_0': [], 'class_1': [], 'class_2': []}
        
        # Track hybrid loss components
        epoch_hybrid_tracking = {'regression': [], 'classification': [], 'class_zeros': [], 'class_ones': [], 'class_middle': []}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Efficient transfer to device
                sequences = batch['sequences'].to(self.device, non_blocking=self.non_blocking)
                splice_labels = batch['splice_labels'].to(self.device, non_blocking=self.non_blocking)
                usage_targets = batch['usage_targets'].to(self.device, non_blocking=self.non_blocking)
                species_ids = batch['species_id'].to(self.device, non_blocking=self.non_blocking)  # NEW
                
                # Forward pass with mixed precision
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    output = self.model(sequences, species_ids=species_ids)
                    
                    # Compute losses (same as training)
                    splice_logits = output['splice_logits']
                    splice_loss = self.splice_criterion(
                        splice_logits.reshape(-1, splice_logits.size(-1)),
                        splice_labels.reshape(-1)
                    )
                    
                    # Track splice class losses
                    splice_class_losses = self._track_splice_class_losses(
                        splice_logits, splice_labels, is_training=False
                    )
                    for class_name, loss_val in splice_class_losses.items():
                        epoch_splice_class[class_name].append(loss_val)
                    
                    splice_mask = (splice_labels > 0)
                    usage_targets = torch.nan_to_num(usage_targets, nan=0.0)
                    
                    # Compute usage loss based on type (SAME AS TRAINING)
                    if self.usage_loss_type == 'hybrid':
                        usage_predictions = output['usage_predictions']
                        usage_class_logits = output.get('usage_class_logits', None)
                        
                        if usage_class_logits is None:
                            raise ValueError("Model must output 'usage_class_logits' for hybrid loss")
                        
                        if splice_mask.sum() > 0:
                            mask_flat = splice_mask.reshape(-1)
                            usage_preds_flat = usage_predictions.reshape(-1, usage_predictions.shape[-1])
                            usage_class_flat = usage_class_logits.reshape(-1, usage_class_logits.shape[-2], usage_class_logits.shape[-1])
                            usage_targets_flat = usage_targets.reshape(-1, usage_targets.shape[-1])
                            
                            # Get component losses
                            original_reduction = self.usage_criterion.reduction
                            self.usage_criterion.reduction = 'none'
                            reg_loss, class_loss = self.usage_criterion.get_component_losses(
                                usage_preds_flat[mask_flat],
                                usage_class_flat[mask_flat],
                                usage_targets_flat[mask_flat]
                            )
                            self.usage_criterion.reduction = original_reduction
                            
                            # Track components
                            epoch_hybrid_tracking['regression'].append(reg_loss.mean().item())
                            epoch_hybrid_tracking['classification'].append(class_loss.mean().item())
                            
                            # Track classification loss by SSE range
                            hybrid_class_losses = self._track_hybrid_class_losses(
                                usage_predictions, usage_class_logits, usage_targets, splice_mask, is_training=False
                            )
                            epoch_hybrid_tracking['class_zeros'].append(hybrid_class_losses['zeros'])
                            epoch_hybrid_tracking['class_ones'].append(hybrid_class_losses['ones'])
                            epoch_hybrid_tracking['class_middle'].append(hybrid_class_losses['middle'])
                            
                            # Combined loss
                            usage_loss = self.usage_criterion(
                                usage_preds_flat[mask_flat],
                                usage_class_flat[mask_flat],
                                usage_targets_flat[mask_flat]
                            )
                            
                            # Track SSE losses
                            sse_tracking = self._track_sse_losses(
                                usage_predictions, usage_targets, splice_mask,
                                usage_class_logits=usage_class_logits,
                                is_training=False
                            )
                            if sse_tracking is not None:
                                epoch_sse_tracking['zeros'].extend(sse_tracking['zeros'].tolist())
                                epoch_sse_tracking['ones'].extend(sse_tracking['ones'].tolist())
                                epoch_sse_tracking['middle'].extend(sse_tracking['middle'].tolist())
                                epoch_sse_tracking['n_zeros'] += sse_tracking['n_zeros']
                                epoch_sse_tracking['n_ones'] += sse_tracking['n_ones']
                                epoch_sse_tracking['n_middle'] += sse_tracking['n_middle']
                                epoch_sse_tracking['targets'].append(sse_tracking['targets'])
                                epoch_sse_tracking['predictions'].append(sse_tracking['predictions'])
                        else:
                            usage_loss = torch.tensor(0.0, device=self.device)
                    elif self.usage_loss_type == 'weighted_mse':
                        usage_predictions = output['usage_predictions']
                        if splice_mask.sum() > 0:
                            mask_flat = splice_mask.reshape(-1)
                            usage_targets_flat = usage_targets.reshape(-1, usage_targets.shape[-1])
                            usage_predictions_flat = usage_predictions.reshape(-1, usage_predictions.shape[-1])
                            usage_loss = self.usage_criterion(
                                usage_predictions_flat[mask_flat],
                                usage_targets_flat[mask_flat]
                            )
                            
                            # Track SSE losses
                            sse_tracking = self._track_sse_losses(
                                usage_predictions, usage_targets, splice_mask,
                                usage_class_logits=None,
                                is_training=False
                            )
                            if sse_tracking is not None:
                                epoch_sse_tracking['zeros'].extend(sse_tracking['zeros'].tolist())
                                epoch_sse_tracking['ones'].extend(sse_tracking['ones'].tolist())
                                epoch_sse_tracking['middle'].extend(sse_tracking['middle'].tolist())
                                epoch_sse_tracking['n_zeros'] += sse_tracking['n_zeros']
                                epoch_sse_tracking['n_ones'] += sse_tracking['n_ones']
                                epoch_sse_tracking['n_middle'] += sse_tracking['n_middle']
                                epoch_sse_tracking['targets'].append(sse_tracking['targets'])
                                epoch_sse_tracking['predictions'].append(sse_tracking['predictions'])
                        else:
                            usage_loss = torch.tensor(0.0, device=self.device)
                    else:
                        usage_predictions = output['usage_predictions']
                        if splice_mask.sum() > 0:
                            mask_flat = splice_mask.reshape(-1)
                            usage_targets_flat = usage_targets.reshape(-1, usage_targets.shape[-1])
                            usage_predictions_flat = usage_predictions.reshape(-1, usage_predictions.shape[-1])
                            usage_loss = self.usage_criterion(
                                usage_predictions_flat[mask_flat],
                                usage_targets_flat[mask_flat]
                            )
                        else:
                            usage_loss = torch.tensor(0.0, device=self.device)
                    
                    loss = (
                        self.splice_weight * splice_loss +
                        self.usage_weight * usage_loss
                    )
                
                # Accumulate losses
                total_loss += loss.item()
                total_splice_loss += splice_loss.item()
                total_usage_loss += usage_loss.item()
                n_batches += 1
                
                # Clear CUDA cache periodically
                if self.device == 'cuda' and batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
        
        # Store splice class tracking
        self.splice_class_tracking['val'][self.current_epoch + 1] = {
            'class_0': np.mean(epoch_splice_class['class_0']) if epoch_splice_class['class_0'] else 0.0,
            'class_1': np.mean(epoch_splice_class['class_1']) if epoch_splice_class['class_1'] else 0.0,
            'class_2': np.mean(epoch_splice_class['class_2']) if epoch_splice_class['class_2'] else 0.0,
        }
        
        # Store hybrid tracking
        if self.usage_loss_type == 'hybrid' and epoch_hybrid_tracking['regression']:
            self.hybrid_loss_tracking['val'][self.current_epoch + 1] = {
                'regression': np.mean(epoch_hybrid_tracking['regression']),
                'classification': np.mean(epoch_hybrid_tracking['classification']),
                'class_zeros': np.mean([x for x in epoch_hybrid_tracking['class_zeros'] if x > 0]) if any(x > 0 for x in epoch_hybrid_tracking['class_zeros']) else 0.0,
                'class_ones': np.mean([x for x in epoch_hybrid_tracking['class_ones'] if x > 0]) if any(x > 0 for x in epoch_hybrid_tracking['class_ones']) else 0.0,
                'class_middle': np.mean([x for x in epoch_hybrid_tracking['class_middle'] if x > 0]) if any(x > 0 for x in epoch_hybrid_tracking['class_middle']) else 0.0
            }
        
        # Store SSE tracking for this epoch
        self.sse_loss_tracking['val'][self.current_epoch + 1] = {
            'zeros': np.array(epoch_sse_tracking['zeros']),
            'ones': np.array(epoch_sse_tracking['ones']),
            'middle': np.array(epoch_sse_tracking['middle']),
            'n_zeros': epoch_sse_tracking['n_zeros'],
            'n_ones': epoch_sse_tracking['n_ones'],
            'n_middle': epoch_sse_tracking['n_middle'],
            'targets': np.concatenate(epoch_sse_tracking['targets']) if epoch_sse_tracking['targets'] else np.array([]),
            'predictions': np.concatenate(epoch_sse_tracking['predictions']) if epoch_sse_tracking['predictions'] else np.array([])
        }
        
        # Log to TensorBoard
        if self.writer is not None:
            epoch_num = self.current_epoch
            
            # Splice class losses
            class_tracking = self.splice_class_tracking['val'][self.current_epoch + 1]
            self.writer.add_scalar('Splice_Loss/Val_Class_0', class_tracking['class_0'], epoch_num)
            self.writer.add_scalar('Splice_Loss/Val_Class_1', class_tracking['class_1'], epoch_num)
            self.writer.add_scalar('Splice_Loss/Val_Class_2', class_tracking['class_2'], epoch_num)
            
            self.writer.add_scalars('Splice_Loss/Val_All_Classes', {
                'class_0': class_tracking['class_0'],
                'class_1': class_tracking['class_1'],
                'class_2': class_tracking['class_2']
            }, epoch_num)
            
            # Hybrid loss components
            if self.usage_loss_type == 'hybrid' and (self.current_epoch + 1) in self.hybrid_loss_tracking['val']:
                hybrid_tracking = self.hybrid_loss_tracking['val'][self.current_epoch + 1]
                self.writer.add_scalar('Hybrid_Loss/Val_Regression', hybrid_tracking['regression'], epoch_num)
                self.writer.add_scalar('Hybrid_Loss/Val_Classification', hybrid_tracking['classification'], epoch_num)
                
                self.writer.add_scalars('Hybrid_Loss/Val_Components', {
                    'regression': hybrid_tracking['regression'],
                    'classification': hybrid_tracking['classification']
                }, epoch_num)
                
                # Classification losses by SSE range
                self.writer.add_scalar('Hybrid_Classification/Val_Zeros', hybrid_tracking['class_zeros'], epoch_num)
                self.writer.add_scalar('Hybrid_Classification/Val_Ones', hybrid_tracking['class_ones'], epoch_num)
                self.writer.add_scalar('Hybrid_Classification/Val_Middle', hybrid_tracking['class_middle'], epoch_num)
                
                self.writer.add_scalars('Hybrid_Classification/Val_By_Range', {
                    'zeros': hybrid_tracking['class_zeros'],
                    'ones': hybrid_tracking['class_ones'],
                    'middle': hybrid_tracking['class_middle']
                }, epoch_num)
            
            # Correlation
            if len(epoch_sse_tracking['targets']) > 0 and len(epoch_sse_tracking['predictions']) > 0:
                targets = np.concatenate(epoch_sse_tracking['targets']).flatten()
                preds = np.concatenate(epoch_sse_tracking['predictions']).flatten()
                correlation = np.corrcoef(targets, preds)[0, 1]
                self.writer.add_scalar('Usage_Metrics/Val_Correlation', correlation, epoch_num)
        
        return {
            'loss': total_loss / n_batches,
            'splice_loss': total_splice_loss / n_batches,
            'usage_loss': total_usage_loss / n_batches
        }
    
    def train(
        self,
        n_epochs: int,
        verbose: bool = True,
        save_best: bool = True,
        early_stopping_patience: Optional[int] = None,
        resume_from: Optional[str] = None
    ):
        """
        Train the model for multiple epochs.
        
        Args:
            n_epochs: Number of epochs to train
            verbose: Whether to print progress
            save_best: Whether to save best model checkpoint
            early_stopping_patience: Stop if no improvement for N epochs
            resume_from: Path to checkpoint file to resume from (or 'auto' to find latest)
        """
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from:
            if resume_from == 'auto':
                # Find latest checkpoint
                resume_from = self._find_latest_checkpoint()
            
            if resume_from and Path(resume_from).exists():
                if verbose:
                    print(f"\nResuming from checkpoint: {resume_from}")
                self.load_checkpoint(resume_from, verbose=verbose)
                start_epoch = self.current_epoch + 1
                if verbose:
                    print(f"Resuming from epoch {start_epoch}")
                    print(f"Best validation loss so far: {self.best_val_loss:.4f}\n")
            elif resume_from != 'auto':
                print(f"Warning: Checkpoint {resume_from} not found, starting from scratch")
        
        epochs_without_improvement = 0
        self.training_start_time = time.time()
        
        for epoch in range(start_epoch, n_epochs):
            self.current_epoch = epoch
            self.epoch_start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Calculate epoch duration
            epoch_duration = time.time() - self.epoch_start_time
            
            # Store first epoch duration for estimation
            if epoch == 0:
                self.first_epoch_duration = epoch_duration
                if verbose:
                    estimated_total = self.first_epoch_duration * n_epochs
                    estimated_end = datetime.now() + timedelta(seconds=estimated_total)
                    print(f"\nFirst epoch completed in {epoch_duration:.1f}s")
                    print(f"Estimated total training time: {timedelta(seconds=int(estimated_total))}")
                    print(f"Estimated completion: {estimated_end.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_splice_loss'].append(train_metrics['splice_loss'])
            self.history['train_usage_loss'].append(train_metrics['usage_loss'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            if val_metrics:
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_splice_loss'].append(val_metrics['splice_loss'])
                self.history['val_usage_loss'].append(val_metrics['usage_loss'])
            
            # Log epoch metrics to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('Train/Loss_Epoch', train_metrics['loss'], epoch)
                self.writer.add_scalar('Train/Splice_Loss_Epoch', train_metrics['splice_loss'], epoch)
                self.writer.add_scalar('Train/Usage_Loss_Epoch', train_metrics['usage_loss'], epoch)
                self.writer.add_scalar('Train/Epoch_Duration', epoch_duration, epoch)
                
                if val_metrics:
                    self.writer.add_scalar('Val/Loss_Epoch', val_metrics['loss'], epoch)
                    self.writer.add_scalar('Val/Splice_Loss_Epoch', val_metrics['splice_loss'], epoch)
                    self.writer.add_scalar('Val/Usage_Loss_Epoch', val_metrics['usage_loss'], epoch)
                
                # Log combined metrics for easy comparison - all under Loss category
                self.writer.add_scalars('Loss/Loss', {
                    'train': train_metrics['loss'],
                    'val': val_metrics['loss'] if val_metrics else 0.0
                }, epoch)
                
                self.writer.add_scalars('Loss/Splice', {
                    'train': train_metrics['splice_loss'],
                    'val': val_metrics['splice_loss'] if val_metrics else 0.0
                }, epoch)
                
                self.writer.add_scalars('Loss/Usage', {
                    'train': train_metrics['usage_loss'],
                    'val': val_metrics['usage_loss'] if val_metrics else 0.0
                }, epoch)
            
            # Print progress with timing
            if verbose:
                elapsed_total = time.time() - self.training_start_time
                
                msg = f"Epoch {epoch+1}/{n_epochs} ({epoch_duration:.1f}s) - "
                msg += f"Train Loss: {train_metrics['loss']:.4f} "
                msg += f"(Splice: {train_metrics['splice_loss']:.4f}, "
                msg += f"Usage: {train_metrics['usage_loss']:.4f})"
                
                if val_metrics:
                    msg += f" - Val Loss: {val_metrics['loss']:.4f} "
                    msg += f"(Splice: {val_metrics['splice_loss']:.4f}, "
                msg += f"Usage: {val_metrics['usage_loss']:.4f})"
                
                # Add ETA based on average epoch time
                if epoch > 0:
                    avg_epoch_time = elapsed_total / (epoch + 1)
                    remaining_epochs = n_epochs - epoch - 1
                    eta_seconds = avg_epoch_time * remaining_epochs
                    eta = datetime.now() + timedelta(seconds=eta_seconds)
                    msg += f" - ETA: {eta.strftime('%H:%M:%S')}"
                
                print(msg)
            
            # Save best model
            if save_best and val_metrics:
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    epochs_without_improvement = 0
                    if self.checkpoint_dir:
                        self.save_checkpoint('best_model.pt')
                        if verbose:
                            print(f"  → Saved best model (val_loss: {self.best_val_loss:.4f})")
                else:
                    epochs_without_improvement += 1
            
            # Early stopping
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping after {epoch+1} epochs")
                break
            
            # Save periodic checkpoint
            if self.checkpoint_dir and (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        # Final summary
        if verbose and self.training_start_time:
            total_time = time.time() - self.training_start_time
            avg_epoch_time = total_time / (epoch + 1)
            print(f"\nTraining Summary:")
            print(f"  Total epochs: {epoch + 1}")
            print(f"  Total time: {timedelta(seconds=int(total_time))}")
            print(f"  Avg epoch time: {avg_epoch_time:.1f}s")
            print(f"  Best val loss: {self.best_val_loss:.4f}")
        
        # Save diagnostic plots
        if self.checkpoint_dir:
            if verbose:
                print(f"\nSaving diagnostic plots...")
            self.save_diagnostic_plots()
            if verbose:
                print(f"  Diagnostic plots saved to: {self.checkpoint_dir / 'diagnostics'}")
        
        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
    
    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint file."""
        if self.checkpoint_dir is None:
            return None
        
        # Look for periodic checkpoints
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        if checkpoints:
            # Sort by epoch number
            checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
            return str(checkpoints[-1])
        
        # Fall back to best model if no periodic checkpoints
        best_model = self.checkpoint_dir / 'best_model.pt'
        if best_model.exists():
            return str(best_model)
        
        return None
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'training_time': time.time() - self.training_start_time if self.training_start_time else 0
        }
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, filename: str, verbose: bool = True):
        """Load model checkpoint."""
        if isinstance(filename, str) and not Path(filename).is_absolute():
            if self.checkpoint_dir is None:
                raise ValueError("checkpoint_dir not set")
            checkpoint_path = self.checkpoint_dir / filename
        else:
            checkpoint_path = Path(filename)
        
        if verbose:
            print(f"Loading checkpoint from {checkpoint_path}...")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        if verbose:
            print(f"  Loaded epoch {self.current_epoch}")
            print(f"  Global step: {self.global_step}")
            print(f"  Best val loss: {self.best_val_loss:.4f}")
            print(f"  Training history: {len(self.history['train_loss'])} epochs")
    
    def save_diagnostic_plots(self, output_dir: Optional[Path] = None):
        """Save diagnostic plots for loss tracking."""
        if output_dir is None:
            if self.checkpoint_dir is None:
                return
            output_dir = self.checkpoint_dir / 'diagnostics'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import matplotlib here to avoid issues if not available
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not available, skipping diagnostic plots")
            return
        
        # Save data as npz for later analysis
        np.savez(
            output_dir / 'sse_loss_tracking.npz',
            train_tracking=self.sse_loss_tracking['train'],
            val_tracking=self.sse_loss_tracking['val']
        )
        
        # Save splice class tracking
        with open(output_dir / 'splice_class_tracking.json', 'w') as f:
            json.dump(self.splice_class_tracking, f, indent=2)
        
        # Save hybrid loss tracking if available
        if self.usage_loss_type == 'hybrid':
            with open(output_dir / 'hybrid_loss_tracking.json', 'w') as f:
                json.dump(self.hybrid_loss_tracking, f, indent=2)
        
        # Plot 1: Loss progression
        self._plot_loss_progression(output_dir)
        
        # Plot 2: Splice class losses
        self._plot_splice_class_losses(output_dir)
        
        # Plot 3: Hybrid loss components (if applicable)
        if self.usage_loss_type == 'hybrid':
            self._plot_hybrid_loss_components(output_dir)
        
        # Plot 4: Weighted MSE SSE losses (if applicable)
        if self.usage_loss_type == 'weighted_mse':
            self._plot_weighted_mse_losses(output_dir)
        
    def _plot_loss_progression(self, output_dir: Path):
        """Plot loss progression across epochs."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Combined loss
        ax = axes[0, 0]
        epochs = np.arange(1, len(self.history['train_loss']) + 1)
        ax.plot(epochs, self.history['train_loss'], marker='o', label='Train', color='blue', alpha=0.7)
        if self.history['val_loss']:
            ax.plot(epochs, self.history['val_loss'], marker='s', label='Val', color='orange', alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Combined Loss', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Plot 2: Splice loss
        ax = axes[0, 1]
        ax.plot(epochs, self.history['train_splice_loss'], marker='o', label='Train', color='blue', alpha=0.7)
        if self.history['val_splice_loss']:
            ax.plot(epochs, self.history['val_splice_loss'], marker='s', label='Val', color='orange', alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Splice Classification Loss', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Plot 3: Usage loss
        ax = axes[1, 0]
        ax.plot(epochs, self.history['train_usage_loss'], marker='o', label='Train', color='blue', alpha=0.7)
        if self.history['val_usage_loss']:
            ax.plot(epochs, self.history['val_usage_loss'], marker='s', label='Val', color='orange', alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Usage Prediction Loss', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Plot 4: Learning rate
        ax = axes[1, 1]
        ax.plot(epochs, self.history['learning_rate'], marker='o', color='green', alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'loss_progression.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_splice_class_losses(self, output_dir: Path):
        """Plot splice site losses for each class across epochs."""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        for split_idx, (split, ax) in enumerate([('train', ax1), ('val', ax2)]):
            tracking = self.splice_class_tracking[split]
            if not tracking:
                ax.text(0.5, 0.5, f'No {split} data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            epochs = sorted(tracking.keys())
            
            class_0_losses = [tracking[e]['class_0'] for e in epochs]
            class_1_losses = [tracking[e]['class_1'] for e in epochs]
            class_2_losses = [tracking[e]['class_2'] for e in epochs]
            
            epochs = np.array(epochs)
            
            ax.plot(epochs, class_0_losses, marker='o', label='Class 0 (No splice)', color='blue', alpha=0.7)
            ax.plot(epochs, class_1_losses, marker='s', label='Class 1 (Acceptor)', color='orange', alpha=0.7)
            ax.plot(epochs, class_2_losses, marker='^', label='Class 2 (Donor)', color='green', alpha=0.7)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title(f'{split.capitalize()} Splice Class Losses', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'splice_class_losses.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_hybrid_loss_components(self, output_dir: Path):
        """Plot hybrid loss components (regression and classification) across epochs."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Overall components (train)
        ax = axes[0, 0]
        tracking = self.hybrid_loss_tracking['train']
        if tracking:
            epochs = sorted(tracking.keys())
            reg_losses = [tracking[e]['regression'] for e in epochs]
            class_losses = [tracking[e]['classification'] for e in epochs]
            epochs = np.array(epochs)
            
            ax.plot(epochs, reg_losses, marker='o', label='Regression', color='blue', alpha=0.7)
            ax.plot(epochs, class_losses, marker='s', label='Classification', color='orange', alpha=0.7)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Train: Hybrid Loss Components', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No train data', ha='center', va='center', transform=ax.transAxes)
        
        # Plot 2: Overall components (val)
        ax = axes[0, 1]
        tracking = self.hybrid_loss_tracking['val']
        if tracking:
            epochs = sorted(tracking.keys())
            reg_losses = [tracking[e]['regression'] for e in epochs]
            class_losses = [tracking[e]['classification'] for e in epochs]
            epochs = np.array(epochs)
            
            ax.plot(epochs, reg_losses, marker='o', label='Regression', color='blue', alpha=0.7)
            ax.plot(epochs, class_losses, marker='s', label='Classification', color='orange', alpha=0.7)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Val: Hybrid Loss Components', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No val data', ha='center', va='center', transform=ax.transAxes)
        
        # Plot 3: Classification by range (train)
        ax = axes[1, 0]
        tracking = self.hybrid_loss_tracking['train']
        if tracking:
            epochs = sorted(tracking.keys())
            zeros_losses = [tracking[e]['class_zeros'] for e in epochs]
            ones_losses = [tracking[e]['class_ones'] for e in epochs]
            middle_losses = [tracking[e]['class_middle'] for e in epochs]
            epochs = np.array(epochs)
            
            ax.plot(epochs, zeros_losses, marker='o', label='SSE ≈ 0 (< 0.05)', color='blue', alpha=0.7)
            ax.plot(epochs, ones_losses, marker='s', label='SSE ≈ 1 (> 0.95)', color='orange', alpha=0.7)
            ax.plot(epochs, middle_losses, marker='^', label='SSE middle (0.05-0.95)', color='green', alpha=0.7)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Classification Loss', fontsize=12)
            ax.set_title('Train: Classification Loss by SSE Range', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No train data', ha='center', va='center', transform=ax.transAxes)
        
        # Plot 4: Classification by range (val)
        ax = axes[1, 1]
        tracking = self.hybrid_loss_tracking['val']
        if tracking:
            epochs = sorted(tracking.keys())
            zeros_losses = [tracking[e]['class_zeros'] for e in epochs]
            ones_losses = [tracking[e]['class_ones'] for e in epochs]
            middle_losses = [tracking[e]['class_middle'] for e in epochs]
            epochs = np.array(epochs)
            
            ax.plot(epochs, zeros_losses, marker='o', label='SSE ≈ 0 (< 0.05)', color='blue', alpha=0.7)
            ax.plot(epochs, ones_losses, marker='s', label='SSE ≈ 1 (> 0.95)', color='orange', alpha=0.7)
            ax.plot(epochs, middle_losses, marker='^', label='SSE middle (0.05-0.95)', color='green', alpha=0.7)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Classification Loss', fontsize=12)
            ax.set_title('Val: Classification Loss by SSE Range', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No val data', ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'hybrid_loss_components.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_weighted_mse_losses(self, output_dir: Path):
        """Plot weighted MSE losses for different SSE ranges."""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        for split_idx, (split, ax) in enumerate([('train', ax1), ('val', ax2)]):
            tracking = self.sse_loss_tracking[split]
            if not tracking:
                ax.text(0.5, 0.5, f'No {split} data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            epochs = sorted(tracking.keys())
            
            zero_means, zero_stds = [], []
            one_means, one_stds = [], []
            middle_means, middle_stds = [], []
            
            for epoch in epochs:
                data = tracking[epoch]
                if len(data['zeros']) > 0:
                    zero_means.append(np.mean(data['zeros']))
                    zero_stds.append(np.std(data['zeros']))
                else:
                    zero_means.append(np.nan)
                    zero_stds.append(np.nan)
                
                if len(data['ones']) > 0:
                    one_means.append(np.mean(data['ones']))
                    one_stds.append(np.std(data['ones']))
                else:
                    one_means.append(np.nan)
                    one_stds.append(np.nan)
                
                if len(data['middle']) > 0:
                    middle_means.append(np.mean(data['middle']))
                    middle_stds.append(np.std(data['middle']))
                else:
                    middle_means.append(np.nan)
                    middle_stds.append(np.nan)
            
            epochs = np.array(epochs)
            
            ax.plot(epochs, zero_means, marker='o', label='SSE ≈ 0 (< 0.05)', color='blue', alpha=0.7)
            ax.plot(epochs, one_means, marker='s', label='SSE ≈ 1 (> 0.95)', color='orange', alpha=0.7)
            ax.plot(epochs, middle_means, marker='^', label='SSE middle (0.05-0.95)', color='green', alpha=0.7)

            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title(f'{split.capitalize()} Weighted MSE Losses', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'weighted_mse_losses.png', dpi=150, bbox_inches='tight')
        plt.close()

