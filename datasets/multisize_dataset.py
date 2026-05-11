"""Multi-size bead tile dataset for reading preprocessed tiles from disk."""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
from typing import List, Optional


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
        return_count: bool = True,
        input_mode: str = "rf",
        manifest_name: str = "manifest.csv",
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.return_count = return_count
        self.input_mode = input_mode

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
        image = image.astype(np.float32) / 255.0  # [0, 1] range

        # Load mask
        mask_path = os.path.join(self.root_dir, row["mask_path"])
        mask = np.load(mask_path)  # [K, H, W] uint8

        # Compute per-class counts
        count = np.array(
            [(mask[k] > 0).sum() for k in range(mask.shape[0])], dtype=np.int64
        )

        sample = {
            "image": torch.from_numpy(image).float(),
            "mask": torch.from_numpy(mask).byte(),
            "count": torch.from_numpy(count).long(),
            "tile_id": str(row["tile_id"]),
            "source_image_id": str(row["source_image_id"]),
            "class_names": self.class_names,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample