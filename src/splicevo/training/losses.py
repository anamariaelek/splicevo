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
        extreme_low_threshold: float = 0.05,
        extreme_high_threshold: float = 0.95,
        extreme_weight: float = 10.0,
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
    """
    
    def __init__(
        self,
        extreme_low_threshold: float = 0.05,
        extreme_high_threshold: float = 0.95,
        class_weight: float = 1.0,
        reg_weight: float = 0.1,
        reduction: str = 'mean'
    ):
        """
        Args:
            extreme_low_threshold: Values below this are classified as "zero"
            extreme_high_threshold: Values above this are classified as "one"
            class_weight: Weight for classification loss
            reg_weight: Weight for regression loss
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.extreme_low = extreme_low_threshold
        self.extreme_high = extreme_high_threshold
        self.class_weight = class_weight
        self.reg_weight = reg_weight
        self.reduction = reduction
        
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(
        self,
        regression_pred: torch.Tensor,
        class_logits: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            regression_pred: Regression predictions for usage values (same shape as target)
            class_logits: Classification logits [batch, ..., 3] for {is_zero, is_one, is_middle}
            target: Target usage values
            mask: Optional mask, 1=include, 0=exclude
            
        Returns:
            Combined classification and regression loss
        """
        # Create classification targets
        is_zero = (target < self.extreme_low).float()
        is_one = (target > self.extreme_high).float()
        is_middle = ((target >= self.extreme_low) & (target <= self.extreme_high)).float()
        
        # Stack to get [batch, ..., 3]
        class_target = torch.stack([is_zero, is_one, is_middle], dim=-1)
        
        # Classification loss (all positions)
        class_loss = self.bce(class_logits, class_target)
        class_loss = class_loss.mean(dim=-1)  # Average over 3 classes
        
        # Regression loss (only on middle values)
        middle_mask = is_middle.bool()
        reg_loss = self.mse(regression_pred, target)
        
        # Zero out regression loss where target is not middle
        reg_loss = torch.where(middle_mask, reg_loss, torch.zeros_like(reg_loss))
        
        # Combine losses
        total_loss = self.class_weight * class_loss + self.reg_weight * reg_loss
        
        # Apply external mask if provided
        if mask is not None:
            total_loss = total_loss * mask
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
