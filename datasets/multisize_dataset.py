"""Multi-size bead tile dataset for reading preprocessed tiles from disk."""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
from skimage.draw import disk
from typing import List, Optional


def center_mask_to_disk_heatmap(center_mask, radius_per_class):
    """Convert center mask to disk heatmap.

    center_mask: [K, H, W], 0/1 float32 or uint8
    radius_per_class: list[int], len=K
    return: [K, H, W], 0.0/1.0 float32
    """
    K, H, W = center_mask.shape
    heatmap = np.zeros_like(center_mask, dtype=np.float32)
    for k in range(K):
        radius = radius_per_class[k]
        if radius <= 0:
            continue
        ys, xs = np.nonzero(center_mask[k] > 0)
        for y, x in zip(ys, xs):
            rr, cc = disk((y, x), radius, shape=(H, W))
            heatmap[k, rr, cc] = 1.0
    return heatmap


class MultiSizeTileManifest:
    """Reads and filters the tile manifest CSV."""

    def __init__(self, root_dir: str, manifest_name: str = "manifest.csv"):
        self.root_dir = root_dir
        self.manifest_path = os.path.join(root_dir, manifest_name)
        self.df = pd.read_csv(self.manifest_path)

    def filter(self, split: str) -> pd.DataFrame:
        """Return rows matching the given split."""
        return self.df[self.df["split"] == split].reset_index(drop=True)

    @property
    def class_names(self) -> List[str]:
        """Return list of class names from first row's class_names column."""
        return self.df.iloc[0]["class_names"].split(",")

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


class MultiSizeBeadTileDataset(Dataset):
    """PyTorch dataset for multi-size bead tiles."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train_pool",
        num_classes: Optional[int] = None,
        transform=None,
        manifest_name: str = "manifest.csv",
        target_mode: str = "raw",
        radius_per_class: Optional[List[int]] = None,
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_mode = target_mode
        self.radius_per_class = radius_per_class or []

        self.manifest = MultiSizeTileManifest(root_dir, manifest_name)
        self.df = self.manifest.filter(split)

        if num_classes is None:
            self.num_classes = self.manifest.num_classes
        else:
            self.num_classes = num_classes

        self.class_names = self.manifest.class_names

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Load image
        image_path = os.path.join(self.root_dir, row["image_path"])
        image = tifffile.imread(image_path)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)  # [1, H, W]
        image = image.astype(np.float32)

        # Load mask: stored as uint8 [0, 255] → normalize to [0.0, 1.0]
        mask_path = os.path.join(self.root_dir, row["mask_path"])
        mask_raw = np.load(mask_path).astype(np.float32) / 255.0  # [K, H, W] float in [0, 1]

        # center_mask: raw 0/1 binary mask, derived from raw storage values
        center_mask = (mask_raw > 0.5).astype(np.float32)  # [K, H, W] 0/1

        # count MUST come from center_mask, not heatmap
        count = np.array(
            [center_mask[k].sum() for k in range(center_mask.shape[0])], dtype=np.int64
        )

        # Determine training mask based on target_mode
        if self.target_mode == "disk":
            # Use disk-ified heatmap as training target
            training_mask = center_mask_to_disk_heatmap(center_mask, self.radius_per_class)
        else:
            # target_mode == "raw": use center_mask directly (current behavior)
            training_mask = center_mask

        sample = {
            "image": torch.from_numpy(image).float(),
            "mask": torch.from_numpy(training_mask).float(),
            "center_mask": torch.from_numpy(center_mask).float(),
            "count": torch.from_numpy(count).long(),
            "tile_id": str(row["tile_id"]),
            "source_image_id": str(row["source_image_id"]),
            "class_names": self.class_names,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
