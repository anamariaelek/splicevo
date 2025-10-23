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
        gradient_accumulation_steps: int = 1
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
        
        # Mixed precision training
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        # Loss functions
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.splice_criterion = nn.CrossEntropyLoss(weight=class_weights)
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
        
        # Time tracking
        self.epoch_start_time = None
        self.first_epoch_duration = None
        self.training_start_time = None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_splice_loss = 0
        total_usage_loss = 0
        n_batches = 0
        
        # Reset gradients at the start
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Efficient transfer to device (non-blocking for memmap data)
            sequences = batch['sequences'].to(self.device, non_blocking=self.non_blocking)
            splice_labels = batch['splice_labels'].to(self.device, non_blocking=self.non_blocking)
            usage_targets = batch['usage_targets'].to(self.device, non_blocking=self.non_blocking)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                output = self.model(sequences)
                
                # Compute splice classification loss
                splice_logits = output['splice_logits']
                splice_loss = self.splice_criterion(
                    splice_logits.reshape(-1, splice_logits.size(-1)),
                    splice_labels.reshape(-1)
                )
                
                # Mask for valid splice positions
                splice_mask = (splice_labels > 0)  # shape: [batch, positions]
                usage_targets = torch.nan_to_num(usage_targets, nan=0.0)
                usage_predictions = output['usage_predictions']
                if splice_mask.sum() > 0:
                    mask_flat = splice_mask.reshape(-1)  # [batch*positions]
                    usage_targets_flat = usage_targets.reshape(-1, usage_targets.shape[-2], usage_targets.shape[-1])  # [batch*positions, n_conditions, 3]
                    usage_predictions_flat = usage_predictions.reshape(-1, usage_predictions.shape[-2], usage_predictions.shape[-1])  # [batch*positions, n_conditions, 3]
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
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Efficient transfer to device
                sequences = batch['sequences'].to(self.device, non_blocking=self.non_blocking)
                splice_labels = batch['splice_labels'].to(self.device, non_blocking=self.non_blocking)
                usage_targets = batch['usage_targets'].to(self.device, non_blocking=self.non_blocking)
                
                # Forward pass with mixed precision (use new torch.amp API)
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    output = self.model(sequences)
                    
                    # Compute losses (same as training)
                    splice_logits = output['splice_logits']
                    splice_loss = self.splice_criterion(
                        splice_logits.reshape(-1, splice_logits.size(-1)),
                        splice_labels.reshape(-1)
                    )
                    
                    splice_mask = (splice_labels > 0).unsqueeze(-1).unsqueeze(-1)
                    usage_predictions = output['usage_predictions']
                    valid_mask = splice_mask & ~torch.isnan(usage_targets)
                    
                    if valid_mask.sum() > 0:
                        usage_loss = self.usage_criterion(
                            usage_predictions[valid_mask],
                            usage_targets[valid_mask]
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
                
                # Log combined metrics for easy comparison
                self.writer.add_scalars('Loss/Combined', {
                    'train': train_metrics['loss'],
                    'val': val_metrics['loss'] if val_metrics else 0.0
                }, epoch)
                
                self.writer.add_scalars('Splice_Loss/Combined', {
                    'train': train_metrics['splice_loss'],
                    'val': val_metrics['splice_loss'] if val_metrics else 0.0
                }, epoch)
                
                self.writer.add_scalars('Usage_Loss/Combined', {
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
