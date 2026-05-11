"""Integration smoke tests for K-class multisize pipeline.

Tests model, loss, and dataset without going through model/__init__.py
(which imports torchvision and has a version conflict).
"""
import torch
import sys
import numpy as np
sys.path.insert(0, '/home/foods/pro/ATTBeadNet')

# Import UNet3Plus directly to avoid torchvision conflict
from model.unet3plus import UNet3Plus
from datasets import MultiSizeBeadTileDataset
from utils.loss import get_loss
from torch.utils.data import DataLoader


def test_unet3plus_output_shape():
    """UNet3Plus outputs correct [B, K, H, W] shape for K=1,2,3,5."""
    for K in [1, 2, 3, 5]:
        model = UNet3Plus(num_classes=K, use_cgm=False, aux_losses=0)
        out = model(torch.randn(2, 1, 64, 64))
        pred = out['final_pred']
        assert pred.shape == (2, K, 64, 64), f"K={K}: expected (2, {K}, 64, 64), got {pred.shape}"
        print(f"K={K}: output shape OK {pred.shape}")


def test_unet3plus_with_cgm():
    """K=1 with use_cgm=True still works."""
    model = UNet3Plus(num_classes=1, use_cgm=True, aux_losses=0)
    out = model(torch.randn(2, 1, 64, 64))
    pred = out['final_pred']
    assert pred.shape == (2, 1, 64, 64)
    print("K=1, use_cgm=True: OK", pred.shape)


def test_loss_focal_dice():
    """focal_dice loss works for K=2."""
    criterion = get_loss("focal_dice")
    logits = torch.randn(4, 2, 64, 64)
    targets = (torch.rand(4, 2, 64, 64) > 0.5).float()
    loss = criterion(logits, targets)
    assert loss.ndim == 0 and loss >= 0
    print(f"FocalDice loss (K=2): {loss.item():.4f}")


def test_loss_focal_dice_k3():
    """focal_dice loss works for K=3."""
    criterion = get_loss("focal_dice")
    logits = torch.randn(2, 3, 32, 32)
    targets = (torch.rand(2, 3, 32, 32) > 0.5).float()
    loss = criterion(logits, targets)
    assert loss.ndim == 0 and loss >= 0
    print(f"FocalDice loss (K=3): {loss.item():.4f}")


def test_dataset_shapes():
    """MultiSizeBeadTileDataset returns correct shapes."""
    ds = MultiSizeBeadTileDataset(
        "/home/foods/pro/ATTBeadNet/datasets/processed/20260511_tiles_64",
        split="train_pool"
    )
    sample = ds[0]
    K = sample["mask"].shape[0]
    print(f"Dataset: image={sample['image'].shape}, mask={sample['mask'].shape}, count={sample['count'].shape}, K={K}")
    assert sample["image"].shape[0] == 1  # single channel
    assert sample["mask"].shape[0] == K   # K classes
    assert sample["count"].shape[0] == K


def test_train_step():
    """One train step forward+backward works end-to-end."""
    ds = MultiSizeBeadTileDataset(
        "/home/foods/pro/ATTBeadNet/datasets/processed/20260511_tiles_64",
        split="train_pool"
    )
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    model = UNet3Plus(num_classes=ds.num_classes, use_cgm=False, aux_losses=0)
    criterion = get_loss("focal_dice")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for batch in loader:
        images = batch["image"].float()
        masks = batch["mask"].float()
        opt.zero_grad()
        logits = model(images)["final_pred"]
        loss = criterion(logits, masks)
        loss.backward()
        opt.step()
        print(f"Train step OK: loss={loss.item():.4f}, K={ds.num_classes}")
        break
