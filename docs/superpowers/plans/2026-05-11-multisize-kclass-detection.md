# Multi-Size K-Class Bead Detection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend UNet3+ for K-class bead center heatmap prediction, with multi-channel loss, per-class validation metrics, and prediction output.

**Architecture:** Reuse existing UNet3+ backbone, replace final `head` Conv2d to output `num_classes` channels. Use independent sigmoid + focal + dice loss per channel. Train on `MultiSizeBeadTileDataset` tiles.

---

## Task 1: Multi-Channel Loss

**Files:**
- Create: `utils/loss/multichannel_loss.py`
- Modify: `utils/loss/__init__.py`

- [ ] **Step 1: Create `utils/loss/multichannel_loss.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossWithLogits(nn.Module):
    """Focal loss for binary segmentation with logits."""

    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class MultiChannelFocalDiceLoss(nn.Module):
    """Sum of Focal + Dice losses across K channels.

    Args:
        class_weights: Optional [K] tensor of per-class weights.
        focal_gamma: Focal loss gamma parameter.
        dice_weight: Weight multiplier for dice component.
        focal_weight: Weight multiplier for focal component.
    """

    def __init__(self, class_weights=None, focal_gamma=2.0, dice_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def dice_per_channel(self, pred, target, smooth=1.0):
        pred = torch.sigmoid(pred)
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        intersection = (pred_flat * target_flat).sum(dim=-1)
        union = pred_flat.sum(dim=-1) + target_flat.sum(dim=-1)
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice  # per-sample, per-channel

    def forward(self, logits, targets, counts=None):
        # logits: [B, K, H, W]
        # targets: [B, K, H, W]  (float 0/1)
        K = logits.shape[1]
        total_loss = 0.0

        for k in range(K):
            pred_k = logits[:, k]
            target_k = targets[:, k]

            # Focal loss
            focal = FocalLossWithLogits(gamma=self.focal_gamma)(pred_k, target_k)

            # Dice loss
            dice = self.dice_per_channel(pred_k, target_k).mean()

            channel_loss = self.focal_weight * focal + self.dice_weight * dice

            if self.class_weights is not None:
                channel_loss = channel_loss * self.class_weights[k]

            total_loss = total_loss + channel_loss

        return total_loss / K
```

- [ ] **Step 2: Add `get_loss` branch in `utils/loss/__init__.py`**

After the existing `get_loss` function, add:
```python
elif loss_function == 'focal_dice':
    from .multichannel_loss import MultiChannelFocalDiceLoss
    criterion = MultiChannelFocalDiceLoss()
```

- [ ] **Step 3: Test the loss**

Create `tests/test_multichannel_loss.py`:
```python
import torch, pytest
from utils.loss.multichannel_loss import MultiChannelFocalDiceLoss, FocalLossWithLogits

def test_focal_dice_loss_shape():
    B, K, H, W = 4, 3, 64, 64
    logits = torch.randn(B, K, H, W)
    targets = (torch.rand(B, K, H, W) > 0.5).float()
    loss = MultiChannelFocalDiceLoss()
    out = loss(logits, targets)
    assert out.ndim == 0
    assert out >= 0

def test_focal_dice_with_weights():
    B, K, H, W = 2, 4, 32, 32
    logits = torch.randn(B, K, H, W)
    targets = torch.zeros(B, K, H, W)
    weights = torch.tensor([1.0, 2.0, 0.5, 1.0])
    loss = MultiChannelFocalDiceLoss(class_weights=weights)
    out = loss(logits, targets)
    assert out.ndim == 0
```

Run: `cd /home/foods/pro/ATTBeadNet && uv run pytest tests/test_multichannel_loss.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add utils/loss/multichannel_loss.py utils/loss/__init__.py tests/test_multichannel_loss.py
git commit -m "feat: add MultiChannelFocalDiceLoss for K-class bead detection"
```

---

## Task 2: Model Head — K-Class Output

**Files:**
- Modify: `model/unet3plus.py`

- [ ] **Step 1: Modify `UNet3Plus.__init__` — remove cls head assumption**

The existing `cls` layer assumes `num_classes <= 2`. For K > 2, replace with `nn.Identity`:

In `UNet3Plus.__init__`, change:
```python
self.cls = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Conv2d(channels[-1], 2, 1),
    nn.AdaptiveMaxPool2d(1),
    nn.Sigmoid()
) if use_cgm and num_classes <= 2 else None
```
to:
```python
self.cls = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Conv2d(channels[-1], 2, 1),
    nn.AdaptiveMaxPool2d(1),
    nn.Sigmoid()
) if use_cgm and num_classes == 1 else None
```

Also change the `if use_cgm and num_classes <= 2` condition to `if use_cgm and num_classes == 1` in the forward pass where `have_obj` is computed.

In `forward()`, change:
```python
if self.cls is not None:
    pred['cls'] = self.cls(de).squeeze_()
    have_obj = torch.argmax(pred['cls'])
```
to:
```python
if self.cls is not None:
    pred['cls'] = self.cls(de).squeeze_()
    have_obj = torch.argmax(pred['cls'])
else:
    have_obj = 1  # always present for multi-class
```

- [ ] **Step 2: Verify output shape**

Run:
```python
import torch
from model import build_unet3plus
for K in [1, 2, 3, 5]:
    model = build_unet3plus(num_classes=K, use_cgm=False)
    out = model(torch.randn(2, 1, 128, 128))
    pred = out['final_pred']
    print(f"K={K}, pred.shape={pred.shape}")  # should be [2, K, 128, 128]
```

- [ ] **Step 3: Commit**

```bash
git add model/unet3plus.py
git commit -m "feat: support arbitrary num_classes in UNet3+ head"
```

---

## Task 3: Config — Add Multisize Training Parameters

**Files:**
- Modify: `config/multisize_bead.yaml`
- Modify: `config/config.py`

- [ ] **Step 1: Add multisize training section to `config/multisize_bead.yaml`**

```yaml
# Multisize K-class bead detection training config

global_seed: 42

data:
  # Data paths
  train_root: "datasets/processed/20260511_tiles_64"
  test_root: ""                    # empty = use internal_val as test
  num_classes: 2
  class_names: ["1.0", "2.8"]
  input_channels: 1
  tile_size: 64
  batch_size: 32
  num_workers: 4

model:
  name: "Unet3"
  encoder: "default"
  skip_ch: 64
  aux_losses: 0                    # disable aux heads for simplicity
  use_cgm: false                   # disable CGM for multi-class
  dropout: 0.3
  pretrained: false

train:
  seed: 42
  epochs: 50
  lr: 0.001
  optimizer: "adamw"
  weight_decay: 0.0001
  val_interval: 1
  device: "cuda"
  loss_type: "focal_dice"
  class_weights: null             # null = uniform, list of K floats for manual weights
  focal_gamma: 2.0
  dice_weight: 1.0
  focal_weight: 1.0
  count_loss_weight: 0.0          # 0 = disabled
  save_name: "multisize_kclass"
  log_dir: "./runs/multisize"

postprocess:
  thresholds: [0.5, 0.5]
  min_distances: [3, 5]
  match_radius: 3
```

- [ ] **Step 2: Update `config/config.py` to handle focal_dice loss and new fields**

Read existing config.py, then add new CN nodes for `postprocess` and extend `train`:
```python
# Add after existing cfg.train entries
cfg.train.class_weights = None
cfg.train.focal_gamma = 2.0
cfg.train.dice_weight = 1.0
cfg.train.focal_weight = 1.0
cfg.train.count_loss_weight = 0.0

# Add postprocess section
cfg.postprocess = CN()
cfg.postprocess.thresholds = [0.5]
cfg.postprocess.min_distances = [3]
cfg.postprocess.match_radius = 3
```

- [ ] **Step 3: Commit**

```bash
git add config/multisize_bead.yaml config/config.py
git commit -m "feat: add multisize K-class config with focal_dice loss and postprocess params"
```

---

## Task 4: Training Loop for Multi-Size Dataset

**Files:**
- Create: `train_multisize.py`

- [ ] **Step 1: Write `train_multisize.py`**

Full training script that:
1. Loads `MultiSizeBeadTileDataset` for train_pool and internal_val
2. Uses `build_unet3plus(num_classes=K)` 
3. Uses `MultiChannelFocalDiceLoss` as criterion
4. Training loop with AMP, per-epoch validation
5. Saves best checkpoint by macro-F1

Key structure (similar to existing `train.py` but for multisize tiles):
```python
# Data loading
train_ds = MultiSizeBeadTileDataset(root_dir=cfg.data.train_root, split="train_pool")
val_ds = MultiSizeBeadTileDataset(root_dir=cfg.data.train_root, split="internal_val")

# For num_classes=1 backward compat: if dataset returns K=1 mask [K,H,W], squeeze to [H,W]
# Model
model = build_unet3plus(
    num_classes=cfg.data.num_classes,
    encoder=cfg.model.encoder,
    skip_ch=cfg.model.skip_ch,
    aux_losses=cfg.model.aux_losses,
    use_cgm=cfg.model.use_cgm,
    pretrained=cfg.model.pretrained,
    dropout=cfg.model.dropout,
)

# Loss
from utils.loss import get_loss
criterion = get_loss(cfg.train.loss_type)

# Training loop
# For each batch: {"image": [B,1,64,64], "mask": [B,K,64,64], ...}
# Forward: logits = model(images)["final_pred"]
# Loss: criterion(logits, masks)
```

**Validation metrics** — accumulate per-class TP, FP, FN across all batches:
```python
# Per-class precision/recall
for k in range(K):
    tp_k = ((pred_k > threshold_k) & (target_k > 0.5)).sum().item()
    fp_k = ((pred_k > threshold_k) & (target_k <= 0.5)).sum().item()
    fn_k = ((pred_k <= threshold_k) & (target_k > 0.5)).sum().item()
    # compute P,R,F1 for class k
```

Save checkpoint as `runs/multisize/multisize_kclass_best.ckpt`.

- [ ] **Step 2: Commit**

```bash
git add train_multisize.py
git commit -m "feat: add train_multisize.py for K-class bead tile training"
```

---

## Task 5: Per-Class Validation Metrics

**Files:**
- Modify: `train_multisize.py` (add `validate` method and per-class metrics)

- [ ] **Step 1: Implement per-class validation**

Add `validate()` function to `Trainer` class in `train_multisize.py`:

```python
def validate(self):
    """Run validation, return per-class metrics dict."""
    self.model.eval()
    device = self.cfg.train.device
    K = self.cfg.data.num_classes
    thresholds = self.cfg.postprocess.thresholds
    class_names = self.cfg.data.class_names

    # Accumulate per-class stats
    tp = [0] * K
    fp = [0] * K
    fn = [0] * K
    count_err = [0.0] * K
    total_count = [0] * K

    with torch.no_grad():
        for batch in self.val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)  # [B, K, H, W]
            counts = batch["count"].to(device)  # [B, K]

            logits = self.model(images)["final_pred"]
            probs = torch.sigmoid(logits)  # [B, K, H, W]

            for k in range(K):
                pred_k = probs[:, k]
                target_k = masks[:, k]
                thresh = thresholds[k] if k < len(thresholds) else 0.5

                tp[k] += ((pred_k > thresh) & (target_k > 0.5)).sum().item()
                fp[k] += ((pred_k > thresh) & (target_k <= 0.5)).sum().item()
                fn[k] += ((pred_k <= thresh) & (target_k > 0.5)).sum().item()

                # Count error
                pred_count_k = (pred_k > thresh).sum(dim=(1, 2)).float()
                gt_count_k = counts[:, k].float()
                count_err[k] += torch.abs(pred_count_k - gt_count_k).sum().item()
                total_count[k] += masks.shape[0]

    # Compute metrics per class
    rows = []
    for k in range(K):
        p = tp[k] / (tp[k] + fp[k]) if (tp[k] + fp[k]) > 0 else 0.0
        r = tp[k] / (tp[k] + fn[k]) if (tp[k] + fn[k]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        mae = count_err[k] / total_count[k] if total_count[k] > 0 else 0.0
        rows.append({
            "class_id": k,
            "class_name": class_names[k],
            "precision": p,
            "recall": r,
            "f1": f1,
            "mae_count": mae,
        })

    # Aggregate
    f1s = [r["f1"] for r in rows]
    macro_f1 = sum(f1s) / len(f1s)
    
    # Log to CSV
    df = pd.DataFrame(rows)
    df.to_csv("val_metrics.csv", index=False)
    
    return {"macro_f1": macro_f1, "per_class": rows}
```

- [ ] **Step 2: Commit**

```bash
git add train_multisize.py
git commit -m "feat: add per-class F1 and count MAE validation to train_multisize"
```

---

## Task 6: Prediction Script — Per-Class Center Extraction

**Files:**
- Create: `predict_multisize.py`

- [ ] **Step 1: Implement `predict_multisize.py`**

```bash
python predict_multisize.py \
  --cfg config/multisize_bead.yaml \
  --ckpt runs/multisize/multisize_kclass_best.ckpt \
  --input-dir data/test_images \
  --out-dir results/multisize \
  --tile-size 64 --stride 32
```

Key functions:

```python
def extract_peaks_per_class(prob_maps, thresholds, min_distances):
    """Extract local maxima peaks per class from probability maps.
    
    prob_maps: np.ndarray [K, H, W]
    thresholds: list of K floats
    min_distances: list of K ints (NMS distance)
    
    Returns: List[Dict] per image, each dict has class_id, x, y, score
    """
    from skimage import measure
    results = []
    K = prob_maps.shape[0]
    for k in range(K):
        prob = prob_maps[k]
        thresh = thresholds[k] if k < len(thresholds) else 0.5
        min_dist = min_distances[k] if k < len(min_distances) else 3
        
        # Threshold
        binary = prob > thresh
        if not binary.any():
            continue
        
        # Connected components
        labeled = measure.label(binary, connectivity=2)
        props = measure.regionprops(labeled, intensity_image=prob)
        
        for prop in props:
            # Use centroid as center
            y, x = prop.centroid
            score = prop.intensity_mean
            results.append({
                "class_id": k,
                "x": int(x),
                "y": int(y),
                "score": float(score)
            })
    
    return results


def predict_tile(model, tile, device):
    """Predict a single tile, return probs [K, H, W]."""
    tile_t = torch.from_numpy(tile).float().unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
    with torch.no_grad():
        logits = model(tile_t)["final_pred"][0]  # [K, H, W]
    return torch.sigmoid(logits).cpu().numpy()


def merge_tile_predictions(tiles_probs, tile_coords, output_shape, tile_size):
    """Merge overlapping tile predictions by averaging probabilities.
    
    tiles_probs: list of [K, tile_size, tile_size] 
    tile_coords: list of (y0, x0) top-left corners
    output_shape: (K, H, W)
    """
    acc = np.zeros(output_shape, dtype=np.float32)
    count = np.zeros(output_shape, dtype=np.float32)
    
    for prob, (y0, x0) in zip(tiles_probs, tile_coords):
        h, w = prob.shape[1], prob.shape[2]
        acc[:, y0:y0+h, x0:x0+w] += prob
        count[:, y0:y0+h, x0:x0+w] += 1
    
    # Avoid division by zero
    count[count == 0] = 1
    return acc / count
```

Main script flow:
1. Load model from checkpoint
2. For each input image: sliding window with tile_size/stride → predict each tile
3. Merge tile predictions → full-image heatmap [K, H, W]
4. `extract_peaks_per_class` → per-class center list
5. Save:
   - `pred_centers.csv` — image_id, class_id, class_name, x, y, score
   - `count_summary.csv` — image_id, count_class_0, count_class_1, ..., total_count
   - `overlays/` — image with colored dots per class
   - `heatmaps/` — per-class heatmap PNG

- [ ] **Step 2: Commit**

```bash
git add predict_multisize.py
git commit -m "feat: add predict_multisize.py with per-class center extraction and CSV output"
```

---

## Task 7: Integration Test — Full Pipeline Smoke Test

**Files:**
- Test on real preprocessed tiles

- [ ] **Step 1: Verify model output shape**

```python
import torch
from model import build_unet3plus
model = build_unet3plus(num_classes=2, use_cgm=False, aux_losses=0)
x = torch.randn(2, 1, 64, 64)
out = model(x)["final_pred"]
assert out.shape == (2, 2, 64, 64), f"Expected (2, 2, 64, 64), got {out.shape}"
print("Model output shape OK:", out.shape)
```

- [ ] **Step 2: Verify loss computation**

```python
import torch
from utils.loss import get_loss
criterion = get_loss("focal_dice")
logits = torch.randn(4, 2, 64, 64)
targets = (torch.rand(4, 2, 64, 64) > 0.5).float()
loss = criterion(logits, targets)
assert loss.ndim == 0 and loss >= 0
print("Loss OK:", loss.item())
```

- [ ] **Step 3: Verify dataset returns correct shapes**

```python
import sys
sys.path.insert(0, "/home/foods/pro/ATTBeadNet")
from datasets import MultiSizeBeadTileDataset
ds = MultiSizeBeadTileDataset("/home/foods/pro/ATTBeadNet/datasets/processed/20260511_tiles_64", split="train_pool")
sample = ds[0]
print("image:", sample["image"].shape)   # [1, 64, 64]
print("mask:", sample["mask"].shape)       # [K, 64, 64]
print("count:", sample["count"].shape)    # [K]
print("K =", sample["mask"].shape[0])
```

- [ ] **Step 4: Run 1 epoch train test (no GPU needed, just verify forward/backward)**

```python
import torch
from model import build_unet3plus
from utils.loss import get_loss
from datasets import MultiSizeBeadTileDataset
from torch.utils.data import DataLoader

ds = MultiSizeBeadTileDataset("/home/foods/pro/ATTBeadNet/datasets/processed/20260511_tiles_64", split="train_pool")
loader = DataLoader(ds, batch_size=8, shuffle=True)
model = build_unet3plus(num_classes=ds.num_classes, use_cgm=False, aux_losses=0)
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
    print("Train step OK, loss =", loss.item())
    break
```

- [ ] **Step 5: Commit smoke test results**

```bash
git add -A
git commit -m "test: integration smoke test for K-class multisize pipeline"
```

---

## File Summary

| File | Action |
|------|--------|
| `utils/loss/multichannel_loss.py` | Create |
| `utils/loss/__init__.py` | Modify |
| `model/unet3plus.py` | Modify |
| `config/multisize_bead.yaml` | Modify |
| `config/config.py` | Modify |
| `train_multisize.py` | Create |
| `predict_multisize.py` | Create |
| `tests/test_multichannel_loss.py` | Create |
