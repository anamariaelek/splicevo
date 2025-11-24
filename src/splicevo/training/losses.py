"""Loss functions for splice site prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE loss where extreme values (near 0 and 1) get higher weights.
    """
    
    def __init__(
        self, 
        extreme_low_threshold: float = 0.1,
        extreme_high_threshold: float = 0.9,
        extreme_weight: float = 5,
        reduction: str = 'mean'
    ):
        """
        Args:
            extreme_low_threshold: Values below this are considered "near zero"
            extreme_high_threshold: Values above this are considered "near one"
            extreme_weight: Weight multiplier for extreme target values
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.extreme_low = extreme_low_threshold
        self.extreme_high = extreme_high_threshold
        self.extreme_weight = extreme_weight
        self.reduction = reduction
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: Predictions
            target: Targets (same shape as pred)
            mask: Optional mask, 1=include, 0=exclude
        """
        # Standard MSE
        mse = (pred - target) ** 2
        
        # Identify extreme targets (near 0 or near 1)
        is_extreme = (target < self.extreme_low) | (target > self.extreme_high)
        
        # Higher weight for extreme targets, 1.0 for everything else
        weights = torch.where(
            is_extreme,
            torch.tensor(self.extreme_weight, device=pred.device, dtype=pred.dtype),
            torch.tensor(1.0, device=pred.device, dtype=pred.dtype)
        )
        
        # Apply weights
        weighted_mse = weights * mse
        
        # Apply mask if provided
        if mask is not None:
            weighted_mse = weighted_mse * mask
            n_elements = mask.sum()
        else:
            n_elements = weighted_mse.numel()
        
        # Reduction
        if self.reduction == 'mean':
            return weighted_mse.sum() / (n_elements + 1e-8)
        elif self.reduction == 'sum':
            return weighted_mse.sum()
        else:  # 'none'
            return weighted_mse


class HybridUsageLoss(nn.Module):
    """
    Hybrid loss for usage prediction combining classification and regression.
    
    For extreme values (near 0 or 1): uses classification loss
    For middle values: uses regression loss
    
    Classification uses CrossEntropyLoss over 3 mutually exclusive classes:
        - Class 0: Low usage (SSE < extreme_low_threshold) - higher weight
        - Class 1: Middle usage (extreme_low_threshold ≤ SSE ≤ extreme_high_threshold)
        - Class 2: High usage (SSE > extreme_high_threshold) - higher weight
    """
    
    def __init__(
        self,
        extreme_low_threshold: float = 0.05,
        extreme_high_threshold: float = 0.95,
        class_weight: float = 1.0,
        reg_weight: float = 0.1,
        extreme_class_weight: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            extreme_low_threshold: Values below this are classified as "low" (class 0)
            extreme_high_threshold: Values above this are classified as "high" (class 2)
            class_weight: Weight for classification loss
            reg_weight: Weight for regression loss
            extreme_class_weight: Weight multiplier for extreme classes vs middle
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        
        # Set instance attributes first
        self.extreme_low_threshold = extreme_low_threshold
        self.extreme_high_threshold = extreme_high_threshold
        self.class_weight = class_weight
        self.reg_weight = reg_weight
        self.extreme_class_weight = extreme_class_weight
        self.reduction = reduction
        
        # Create class weights: [low, middle, high]
        # Give higher weight to extreme classes (0 and 2)
        # Register as buffer so it moves to the correct device automatically
        self.register_buffer(
            'ce_class_weights',
            torch.tensor([extreme_class_weight, 1.0, extreme_class_weight])
        )
        
        # Note: We'll create CE loss in forward pass to ensure weights are on correct device
        self.mse = nn.MSELoss(reduction='none')
    
    def get_component_losses(
        self,
        regression_pred: torch.Tensor,
        class_logits: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ):
        """
        Get classification and regression losses separately.
        
        Args:
            regression_pred: Regression predictions (batch, n_conditions)
            class_logits: Classification logits (batch, n_conditions, 3)
            target: Target values (batch, n_conditions)
            mask: Optional mask (batch, n_conditions)
        
        Returns:
            Tuple of (regression_loss, classification_loss)
        """
        # Create classification targets (class indices)
        # Class 0: low (< extreme_low_threshold)
        # Class 1: middle (>= extreme_low_threshold and <= extreme_high_threshold)
        # Class 2: high (> extreme_high_threshold)
        class_target = torch.zeros_like(target, dtype=torch.long)
        class_target[target < self.extreme_low_threshold] = 0
        class_target[(target >= self.extreme_low_threshold) & (target <= self.extreme_high_threshold)] = 1
        class_target[target > self.extreme_high_threshold] = 2
        
        # Classification loss
        # class_logits: (batch, n_conditions, 3)
        # class_target: (batch, n_conditions)
        # CrossEntropyLoss expects (N, C) and (N), so we need to reshape
        batch_size, n_conditions, n_classes = class_logits.shape
        class_logits_2d = class_logits.reshape(-1, n_classes)  # (batch*n_conditions, 3)
        class_target_1d = class_target.reshape(-1)  # (batch*n_conditions)
        
        # Ensure class weights are on the same device as inputs
        ce_weights = self.ce_class_weights.to(class_logits.device)
        
        # Use functional form with weights on correct device
        class_loss = F.cross_entropy(
            class_logits_2d, 
            class_target_1d, 
            weight=ce_weights,
            reduction='none'
        )  # (batch*n_conditions)
        class_loss = class_loss.reshape(batch_size, n_conditions)  # (batch, n_conditions)
        
        # Regression loss (all positions)
        # Clamp predictions to [0, 1] to prevent extreme values
        regression_pred_clamped = torch.clamp(regression_pred, 0.0, 1.0)
        reg_loss = self.mse(regression_pred_clamped, target)
        
        # Apply mask if provided
        if mask is not None:
            class_loss = class_loss * mask
            reg_loss = reg_loss * mask
        
        return reg_loss, class_loss
    
    def forward(
        self,
        regression_pred: torch.Tensor,
        class_logits: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            regression_pred: Regression predictions for usage values (batch, n_conditions)
            class_logits: Classification logits (batch, n_conditions, 3) for {low, middle, high}
            target: Target usage values (batch, n_conditions)
            mask: Optional mask, 1=include, 0=exclude (batch, n_conditions)
            
        Returns:
            Combined classification and regression loss
        """
        reg_loss, class_loss = self.get_component_losses(
            regression_pred, class_logits, target, mask
        )
        
        # Combine losses
        total_loss = self.class_weight * class_loss + self.reg_weight * reg_loss
        
        # Apply external mask if provided (already applied in get_component_losses)
        if mask is not None:
            n_elements = mask.sum()
        else:
            n_elements = total_loss.numel()
        
        # Reduction
        if self.reduction == 'mean':
            return total_loss.sum() / (n_elements + 1e-8)
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss


def get_usage_loss_fn(loss_type: str = 'weighted_mse', **kwargs) -> nn.Module:
    """
    Factory function to create usage loss functions.
    
    Args:
        loss_type: 'mse', 'weighted_mse', 'hybrid'
        **kwargs: Additional arguments for the loss function
    """
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'weighted_mse':
        return WeightedMSELoss(**kwargs)
    elif loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'hybrid':
        return HybridUsageLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
