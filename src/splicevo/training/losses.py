"""Loss functions for splice site prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in splice site prediction.
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Where:
        - p_t is the probability of the correct class
        - alpha balances positive/negative examples
        - gamma focuses on hard-to-classify examples (gamma=0 is cross-entropy)
    
    This loss is particularly effective for:
    - Extreme class imbalance (e.g., non-splice sites >> splice sites)
    - Reducing influence of easy negative examples
    - Focusing training on hard-to-classify examples
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Weight for each class. Tensor of shape (num_classes,)
                   Default None means equal weight for all classes.
                   Typical usage: [0.25, 1.0, 1.0] to reduce weight on class 0
            gamma: Focusing parameter. Higher values give more weight to hard examples.
                   gamma=0 is equivalent to standard cross-entropy.
                   Recommended values:
                   - gamma=2.0 (default): Good starting point
                   - gamma=3.0-5.0: For extreme imbalance
            reduction: 'mean', 'sum', or 'none'
        
        Example:
            # For splice site prediction with imbalanced classes
            focal_loss = FocalLoss(
                alpha=torch.tensor([0.25, 1.0, 1.0]),  # Reduce weight on class 0
                gamma=2.0  # Focus on hard examples
            )
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Model predictions (logits) of shape (N, C) where C is num_classes
            targets: Ground truth labels of shape (N,) with values in [0, C-1]
        
        Returns:
            Focal loss value
        """
        # Get cross-entropy loss for each example
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get probability of correct class: p_t = exp(-ce_loss)
        p_t = torch.exp(-ce_loss)
        
        # Apply focal term: (1 - p_t)^gamma
        # When p_t is high (confident, easy examples) -> focal_term is small -> reduced loss
        # When p_t is low (uncertain, hard examples) -> focal_term is large -> increased loss
        focal_term = (1 - p_t) ** self.gamma
        focal_loss = focal_term * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            # Ensure alpha is on the same device as inputs
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            # Select alpha value for each target's class
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


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
    
    Classification uses CrossEntropyLoss over 3 mutually exclusive classes:
        - Class 0: Low usage (SSE < extreme_low_threshold)
        - Class 1: Middle usage (extreme_low_threshold ≤ SSE ≤ extreme_high_threshold)
        - Class 2: High usage (SSE > extreme_high_threshold)

    Regression uses MSELoss over all positions.
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
            extreme_low_threshold: Values below this are classified as "low"
            extreme_high_threshold: Values above this are classified as "high"
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
        # Give higher weight to extreme classes (low and high)
        # Register as buffer so it moves to the correct device automatically
        self.register_buffer(
            'ce_class_weights',
            torch.tensor([extreme_class_weight, 1.0, extreme_class_weight])
        )
        
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
        reg_loss = self.mse(regression_pred, target)
        
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


class MaskedFocalLoss(nn.Module):
    """
    Focal loss that only computes loss within a context window around positive sites.
    
    This is useful when all sequences contain positive examples but most positions
    are negative. By masking distant negatives, we focus learning on the relevant
    regions around splice sites.
    
    Example:
        >>> loss_fn = MaskedFocalLoss(
        ...     focal_loss=FocalLoss(gamma=2.0, alpha=torch.tensor([0.25, 1.0, 1.0])),
        ...     context_window=100  # Only compute loss within ±100bp of splice sites
        ... )
    """
    
    def __init__(
        self,
        focal_loss: FocalLoss,
        context_window: int = 100
    ):
        """
        Args:
            focal_loss: Base focal loss to use
            context_window: Number of positions around each positive site to include
        """
        super().__init__()
        self.focal_loss = focal_loss
        self.context_window = context_window
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Model predictions of shape (batch, seq_len, num_classes)
            targets: Ground truth labels of shape (batch, seq_len)
        
        Returns:
            Masked focal loss value
        """
        if inputs.dim() == 2:
            # Already flattened (N, C)
            return self.focal_loss(inputs, targets)
        
        batch_size, seq_len, num_classes = inputs.shape
        
        # Create mask for positions to include in loss
        loss_mask = torch.zeros_like(targets, dtype=torch.bool)
        
        for b in range(batch_size):
            # Find positive positions
            pos_indices = torch.where(targets[b] > 0)[0]
            
            if len(pos_indices) > 0:
                # Include all positives
                loss_mask[b, pos_indices] = True
                
                # Include negatives within window of positives
                for pos_idx in pos_indices:
                    start = max(0, pos_idx - self.context_window)
                    end = min(seq_len, pos_idx + self.context_window + 1)
                    loss_mask[b, start:end] = True
            else:
                # If no positives (shouldn't happen), include all positions
                loss_mask[b, :] = True
        
        # Flatten and apply mask
        inputs_flat = inputs.reshape(-1, num_classes)
        targets_flat = targets.reshape(-1)
        loss_mask_flat = loss_mask.reshape(-1)
        
        # Compute loss only on masked positions
        if loss_mask_flat.sum() > 0:
            masked_inputs = inputs_flat[loss_mask_flat]
            masked_targets = targets_flat[loss_mask_flat]
            
            loss = self.focal_loss(masked_inputs, masked_targets)
            return loss
        else:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)


class HardNegativeMiningLoss(nn.Module):
    """
    Focus training on hard negative examples.
    
    Instead of using all negative positions, this loss identifies the most difficult
    negatives (those with high predicted probability) and only trains on those plus
    all positive examples.
    
    This is effective when you have many easy negatives that dominate the loss,
    preventing the model from learning the harder cases.
    
    Example:
        >>> loss_fn = HardNegativeMiningLoss(
        ...     focal_loss=FocalLoss(gamma=2.0),
        ...     negative_ratio=3.0  # Use 3 negatives for every positive
        ... )
    """
    
    def __init__(
        self,
        focal_loss: FocalLoss,
        negative_ratio: float = 3.0
    ):
        """
        Args:
            focal_loss: Base focal loss to use
            negative_ratio: Ratio of negative to positive examples to keep
                          E.g., 3.0 means keep 3 negatives for each positive
        """
        super().__init__()
        self.focal_loss = focal_loss
        self.negative_ratio = negative_ratio
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Model predictions of shape (N, C) or (batch, seq_len, num_classes)
            targets: Ground truth labels of shape (N,) or (batch, seq_len)
        
        Returns:
            Loss computed on positives + hard negatives
        """
        # Flatten if needed
        if inputs.dim() == 3:
            batch_size, seq_len, num_classes = inputs.shape
            inputs = inputs.reshape(-1, num_classes)
            targets = targets.reshape(-1)
        
        # Separate positive and negative examples
        pos_mask = targets > 0
        neg_mask = targets == 0
        
        n_positives = pos_mask.sum().item()
        
        if n_positives == 0:
            # No positives, use all negatives
            return self.focal_loss(inputs, targets)
        
        # Always include all positives
        pos_inputs = inputs[pos_mask]
        pos_targets = targets[pos_mask]
        
        # Compute loss for all negatives to identify hard ones
        neg_inputs = inputs[neg_mask]
        neg_targets = targets[neg_mask]
        
        if len(neg_inputs) == 0:
            # No negatives (shouldn't happen)
            return self.focal_loss(pos_inputs, pos_targets)
        
        # Get negative loss for each example
        with torch.no_grad():
            neg_losses = F.cross_entropy(neg_inputs, neg_targets, reduction='none')
        
        # Select hard negatives (highest loss = most difficult)
        n_hard_negatives = int(n_positives * self.negative_ratio)
        n_hard_negatives = min(n_hard_negatives, len(neg_losses))
        
        if n_hard_negatives > 0:
            hard_neg_indices = torch.topk(neg_losses, n_hard_negatives).indices
            hard_neg_inputs = neg_inputs[hard_neg_indices]
            hard_neg_targets = neg_targets[hard_neg_indices]
            
            # Combine positives and hard negatives
            combined_inputs = torch.cat([pos_inputs, hard_neg_inputs], dim=0)
            combined_targets = torch.cat([pos_targets, hard_neg_targets], dim=0)
        else:
            combined_inputs = pos_inputs
            combined_targets = pos_targets
        
        # Compute loss on selected examples
        loss = self.focal_loss(combined_inputs, combined_targets)
        return loss


def get_splice_loss_fn(loss_type: str = 'focal', **kwargs) -> nn.Module:
    """
    Factory function to create splice site loss functions.
    
    Args:
        loss_type: 'cross_entropy', 'focal', 'masked_focal', 'hard_negative_mining'
        **kwargs: Additional arguments for the loss function
    """
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'masked_focal':
        base_focal = FocalLoss(**kwargs.get('focal_params', {}))
        context_window = kwargs.get('context_window', 100)
        return MaskedFocalLoss(focal_loss=base_focal, context_window=context_window)
    elif loss_type == 'hard_negative_mining':
        base_focal = FocalLoss(**kwargs.get('focal_params', {}))
        negative_ratio = kwargs.get('negative_ratio', 3.0)
        return HardNegativeMiningLoss(focal_loss=base_focal, negative_ratio=negative_ratio)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


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
