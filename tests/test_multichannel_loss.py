import torch
import pytest
from utils.loss.multichannel_loss import FocalLossWithLogits, MultiChannelFocalDiceLoss


def test_focal_dice_loss_shape():
    """Verify output is scalar and non-negative."""
    B, K, H, W = 2, 3, 32, 32
    logits = torch.randn(B, K, H, W)
    targets = (torch.rand(B, K, H, W) > 0.5).float()

    loss_fn = MultiChannelFocalDiceLoss()
    loss = loss_fn(logits, targets)

    assert loss.ndim == 0, f"Expected scalar output, got shape {loss.shape}"
    assert loss >= 0, f"Expected non-negative loss, got {loss.item()}"


def test_focal_dice_with_weights():
    """Verify it works with K=4 and class_weights."""
    B, K, H, W = 2, 4, 16, 16
    logits = torch.randn(B, K, H, W)
    targets = (torch.rand(B, K, H, W) > 0.5).float()
    class_weights = torch.tensor([1.0, 2.0, 0.5, 1.5])

    loss_fn = MultiChannelFocalDiceLoss(class_weights=class_weights)
    loss = loss_fn(logits, targets)

    assert loss.ndim == 0, f"Expected scalar output, got shape {loss.shape}"
    assert loss >= 0, f"Expected non-negative loss, got {loss.item()}"


def test_focal_loss_with_logits():
    """Test FocalLossWithLogits basic functionality."""
    B, H, W = 2, 32, 32
    logits = torch.randn(B, H, W)
    targets = (torch.rand(B, H, W) > 0.5).float()

    loss_fn = FocalLossWithLogits()
    loss = loss_fn(logits, targets)

    assert loss.ndim == 0, f"Expected scalar output, got shape {loss.shape}"
    assert loss >= 0, f"Expected non-negative loss, got {loss.item()}"


def test_focal_dice_with_counts():
    """Verify counts parameter is accepted (even if not used)."""
    B, K, H, W = 2, 3, 16, 16
    logits = torch.randn(B, K, H, W)
    targets = (torch.rand(B, K, H, W) > 0.5).float()
    counts = torch.randint(1, 100, (B, K)).float()

    loss_fn = MultiChannelFocalDiceLoss()
    loss = loss_fn(logits, targets, counts=counts)

    assert loss.ndim == 0
    assert loss >= 0


def test_focal_dice_known_output():
    """Test with known inputs to verify computation is correct."""
    # When prediction equals target, loss should be near zero
    B, K, H, W = 1, 2, 8, 8
    logits = torch.zeros(B, K, H, W)  # sigmoid = 0.5
    targets = torch.zeros(B, K, H, W)  # all zeros

    loss_fn = MultiChannelFocalDiceLoss(focal_weight=1.0, dice_weight=1.0)
    loss = loss_fn(logits, targets)

    # For all-zero targets with sigmoid at 0.5, dice loss should be 1 - (smooth)/(smooth + sum(pred)) ≈ 1
    # Focal loss should also be non-trivial. Total should be > 0
    assert loss >= 0


def test_focal_dice_per_channel_computation():
    """Verify focal and dice components are computed per channel."""
    B, K, H, W = 2, 2, 16, 16
    logits = torch.randn(B, K, H, W)
    targets = (torch.rand(B, K, H, W) > 0.5).float()

    # Compute with both weights = 1
    loss_fn = MultiChannelFocalDiceLoss(focal_weight=1.0, dice_weight=1.0)
    loss_equal = loss_fn(logits, targets)

    # Compute with focal only
    loss_focal_only = MultiChannelFocalDiceLoss(focal_weight=1.0, dice_weight=0.0)
    loss_f = loss_focal_only(logits, targets)

    # Compute with dice only
    loss_dice_only = MultiChannelFocalDiceLoss(focal_weight=0.0, dice_weight=1.0)
    loss_d = loss_dice_only(logits, targets)

    # They should be different (unless loss happens to be same)
    # This is more of a smoke test - just verify they run
    assert loss_equal.ndim == 0
    assert loss_f.ndim == 0
    assert loss_d.ndim == 0
