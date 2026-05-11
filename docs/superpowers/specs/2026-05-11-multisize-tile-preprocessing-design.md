# Multi-Size Bead Tile Preprocessing Pipeline — Design Spec

## 1. Overview

A reusable, offline preprocessing pipeline that:
1. Reads raw RF images + per-class center masks from `datasets/2026XXXX/{RF, 1.0_mask, 2.8_mask, ...}`
2. Tiles them into fixed-size (64×64) patches with synchronized K-channel center masks
3. Applies seeded deterministic D4 geometric augmentation to training tiles
4. Splits into train/val/test sets by image_id (no tile-level leakage)
5. Writes a standardized output directory with manifest.csv and config.json

## 2. Data Format

### Input (Format A — current real format)

```
datasets/
    20260511/
        RF/
            1.tif, 2.tif, ... 7.tif
        1.0_mask/
            1_Mask.tif, ... 7_Mask.tif
        2.8_mask/
            1_Mask.tif, ... 7_Mask.tif
```

- `RF/{id}.tif` → source image
- `{class_dir}/{id}_Mask.tif` → center mask for that class
- `mask[y, x] == 1` means "a bead center"; `0` means "not a center"
- `class_dir` naming: `{class_name}_mask`, e.g. `1.0_mask` → class name `1.0`

### Output Directory Structure

```
datasets/processed/
    {dataset_name}_tiles_{tile_size}/
        train_pool/
            images/      # .tif files
            masks/       # .npy files [K, H, W]
        internal_val/
            images/
            masks/
        independent_test/
            images/
            masks/
        manifest.csv
        config.json
        debug/
            tiles_overlay/
```

## 3. Tile Filename Convention

```
{source_dataset}_{image_id}_y{y0}_x{x0}[_aug{aug_id}].tif
{source_dataset}_{image_id}_y{y0}_x{x0}[_aug{aug_id}]_mask.npy
```

- `y0`, `x0` are the top-left corner coordinates in the source image
- `aug_id=0` means original (no augmentation suffix written); `aug_id=1..N` for augmented copies
- mask dtype = `uint8`, shape = `[K, tile_size, tile_size]`

## 4. D4 Geometric Transforms

8 unique D4 transforms applied to both image `[H, W]` and mask `[K, H, W]`
using `axes=(-2, -1)` to always operate on the spatial axes regardless of mask channel position:

| aug_id | Name | Image | Mask |
|--------|------|-------|------|
| 0 | identity | as-is | as-is |
| 1 | rot90 | `np.rot90(x, k=1, axes=(-2,-1))` | `np.rot90(x, k=1, axes=(-2,-1))` |
| 2 | rot180 | `np.rot90(x, k=2, axes=(-2,-1))` | `np.rot90(x, k=2, axes=(-2,-1))` |
| 3 | rot270 | `np.rot90(x, k=3, axes=(-2,-1))` | `np.rot90(x, k=3, axes=(-2,-1))` |
| 4 | flip_h | `np.flipud(x)` after 180°rot? No — `np.fliplr(x)` | `np.fliplr(x)` |
| 5 | flip_v | `np.flipud(x)` | `np.flipud(x)` |
| 6 | transpose | `np.transpose(x, axes=(..., 1, 0))` | `np.transpose(x, axes=(..., 1, 0))` |
| 7 | anti_transpose | `np.flipud(np.transpose(x))` | same |

**Note**: For masks with shape `[K, H, W]`, `axes=(-2, -1)` ensures spatial axes are always the last two, so:
- `np.rot90(mask, k=1, axes=(-2,-1))` rotates the spatial dimensions correctly
- `np.fliplr(mask)` and `np.flipud(mask)` work element-wise, no axes needed
- `np.transpose(mask, axes=(0, 2, 1))` swaps H and W for `[K, H, W]`

## 5. Augmentation Multiplier

- `--augment-multiplier N` → generate N augmented copies per original tile
- Original tile always kept as `aug_id=0` (identity)
- `aug_id=1..N` are augmented copies
- For N ≥ 8: sample without replacement from the 7 non-identity D4 transforms
- For N < 8: sample without replacement from the 7 non-identity transforms
- A deterministic RNG seeded by `global_seed + tile_id + aug_id` ensures reproducibility

**Seed derivation:**
```python
seed = hash(global_seed, tile_id, aug_id)  # produces deterministic int
rng = np.random.default_rng(seed)
aug_geom_id = rng.integers(0, 8)  # sample transform
```

## 6. Intensity Normalization

- Percentile clipping: clip to `[p1, p99.8]` then rescale to `[0, 1]`
- Convert image to `float32`
- Mask stays `uint8` with values `0` or `1`

## 7. Sliding Window

- `tile_size=64` (default), configurable
- `stride` configurable: 32, 48, or 64
- Edge handling: skip tiles that extend beyond image boundary (no padding)
- For each tile: compute `count_class_k = int((mask_tile[k] > 0).sum())`

## 8. Filtering

- **Positive tile**: `is_positive = mask_tile.sum() > 0` (any class has a center)
- **Negative tile**: `is_positive = False`
- Keep **all** positive tiles
- Keep negative tiles with probability `negative_ratio` (default 1.0 = keep all)
- `count_total = sum(count_class_k for all k)`

## 9. Split by Image-ID (Before Tiling)

```
1. collect all image_ids present in RF/ with all required masks
2. if split_mode == all_train:
       train_ids = all image_ids
       val_ids = []
       test_ids = []
   elif split_mode == train_val_by_image:
       shuffle image_ids with global_seed
       split at val_ratio
       train_ids, val_ids = split
   elif split_mode == fixed_test:
       train_ids = []
       val_ids = []
       test_ids = all image_ids
3. tile each image_id's image + mask_stack
4. assign tiles to train_pool / internal_val / independent_test based on source image_id
5. only train_pool tiles receive offline augmentation
```

## 10. Offline Augmentation Scope

- **train_pool**: full offline augmentation (original + N augmented copies)
- **internal_val**: original tiles only (no augmentation), used to tune hyperparameters
- **independent_test**: original tiles only (no augmentation), used for final evaluation

If `--augment-offline` is not set: no augmentation is applied to any split.

## 11. Manifest Schema

Columns:

```
tile_id, source_dataset, source_image_id, source_image_path,
split, image_path, mask_path, y0, x0, tile_size, stride,
class_names, count_class_0, count_class_1, ..., count_total,
is_positive, augmentation_id, geom_transform,
source_tile_id, is_augmented
```

- `tile_id`: `{source_dataset}_{image_id}_y{y0}_x{x0}_aug{aug_id}`
- `source_tile_id`: `{source_dataset}_{image_id}_y{y0}_x{x0}` (original, before augmentation)
- `geom_transform`: name string e.g. `"rot90"`, `"identity"`
- `is_augmented`: `True` if `augmentation_id > 0`, else `False`
- `class_names`: comma-separated string e.g. `"1.0,2.8"`

## 12. Config JSON

```json
{
  "global_seed": 42,
  "data": {
    "raw_root": "datasets/20260511",
    "image_dir": "RF",
    "class_dirs": ["1.0_mask", "2.8_mask"],
    "class_names": ["1.0", "2.8"],
    "mask_suffix": "_Mask.tif",
    "image_suffix": ".tif",
    "processed_root": "datasets/processed/20260511_tiles_64",
    "tile_size": 64,
    "stride": 32,
    "split_mode": "all_train",
    "val_ratio": 0.15,
    "negative_ratio": 1.0,
    "mask_center_mode": "pixel"
  },
  "augment": {
    "offline": true,
    "multiplier": 4,
    "geom_transforms": ["identity","rot90","rot180","rot270","flip_h","flip_v","transpose","anti_transpose"],
    "photometric": false
  }
}
```

## 13. Dataset Class API

```python
class MultiSizeBeadTileDataset(Dataset):
    def __init__(self, root_dir, split="train_pool",
                 num_classes=None, transform=None,
                 return_count=True, input_mode="rf",
                 manifest_name="manifest.csv"):
        # loads manifest.csv filtered by split
        # infers num_classes from mask shape
    def __getitem__(self, idx):
        return {
            "image": Tensor[float32, C, H, W],   # [0, 1]
            "mask":  Tensor[uint8, K, H, W],
            "count": Tensor[int64, K],
            "tile_id": str,
            "source_image_id": str,
            "class_names": List[str],
        }
```

- `image`: C=1 for RF (single channel)
- `mask`: `mask[k, y, x] = 1` means a center of class k
- `count[k] = int((mask[k] > 0).sum())`

## 14. Module Map

| File | Responsibility |
|------|----------------|
| `scripts/preprocess_multisize_tiles.py` | CLI, end-to-end pipeline orchestration |
| `datasets/multisize_dataset.py` | `MultiSizeBeadTileDataset` + `MultiSizeTileManifest` |
| `utils/tile_utils.py` | sliding window, padding, overlay debug |
| `utils/augment_multisize.py` | D4 geometric transforms, seeded augmentation |
| `config/multisize_bead.yaml` | default config (all parameters) |

## 15. Debug Overlay

- First 16 tiles from train_pool → overlay saved to `debug/tiles_overlay/`
- Overlay: image grayscale + colored center dots per class
  - class 0 / 1.0: cyan `(0, 255, 255)`
  - class 1 / 2.8: yellow `(255, 255, 0)`
  - class k: HSV color wheel mapping `h = k / K * 360`
- Tiles with no centers: skip or mark with a border

## 16. Edge Cases

- Missing mask for an image_id in any class_dir → warn, skip that image_id
- Image/mask shape mismatch → error, skip that image_id
- No positive tiles from an image → warn but still generate tiles per negative_ratio
- `independent_test` with `augment_offline=true` → error (must be False)
- `negative_ratio=0` → keep zero negative tiles
- `val_ratio=0` or `all_train` mode → `internal_val/` directory may be empty but should be created
