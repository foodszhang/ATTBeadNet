import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossWithLogits(nn.Module):
    """Focal loss for binary segmentation with logits.

    Args:
        alpha (float): Weighting factor in range (0,1). Default: 0.5
        gamma (float): Focusing parameter for modulating loss. Default: 2.0
    """

    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, H, W] or [B, K, H, W] raw model output
            targets: [B, H, W] or [B, K, H, W] float tensor with values 0.0 or 1.0

        Returns:
            Scalar loss
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)  # probability of correct class
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha weighting: higher alpha -> higher weight on positive class
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_weight * focal_weight * bce_loss
        return loss.mean()


class MultiChannelFocalDiceLoss(nn.Module):
    """Multi-channel Focal + Dice loss for binary segmentation.

    Computes per-channel: focal_weight * focal_loss + dice_weight * dice_loss
    Then averages across K channels.

    Args:
        focal_weight (float): Weight for focal loss component. Default: 1.0
        dice_weight (float): Weight for dice loss component. Default: 1.0
        focal_alpha (float): Alpha parameter for focal loss. Default: 0.5
        focal_gamma (float): Gamma parameter for focal loss. Default: 2.0
        class_weights (torch.Tensor, optional): Per-class weights [K]. Default: None
    """

    def __init__(self, focal_weight=1.0, dice_weight=1.0, focal_alpha=0.5, focal_gamma=2.0, class_weights=None):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_loss = FocalLossWithLogits(alpha=focal_alpha, gamma=focal_gamma)
        self.class_weights = class_weights

    def forward(self, logits, targets, counts=None):
        """
        Args:
            logits: [B, K, H, W] raw model output
            targets: [B, K, H, W] float tensor with values 0.0 or 1.0
            counts: optional [B, K] per-class bead counts (not used in v1)

        Returns:
            Scalar loss
        """
        B, K, H, W = logits.shape

        total_loss = 0.0
        for k in range(K):
            # Extract single channel [B, H, W]
            logit_k = logits[:, k, :, :]
            target_k = targets[:, k, :, :]

            # Focal loss for channel k
            focal_k = self.focal_loss(logit_k, target_k)

            # Dice loss for channel k
            pred_k = torch.sigmoid(logit_k)
            smooth = 1.0
            intersection = torch.sum(target_k * pred_k)
            dice_k = 1 - (2.0 * intersection + smooth) / (torch.sum(target_k) + torch.sum(pred_k) + smooth)

            # Combine focal and dice
            channel_loss = self.focal_weight * focal_k + self.dice_weight * dice_k

            # Apply class weight if provided
            if self.class_weights is not None:
                channel_loss = channel_loss * self.class_weights[k]

            total_loss = total_loss + channel_loss

        # Average across channels
        loss = total_loss / K

        return loss
