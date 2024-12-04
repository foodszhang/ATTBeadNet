import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms.functional as F
import skimage as ski
from pathlib import Path
import numpy as np
from typing import Tuple
import os
from skimage import measure
import cv2


def seed_detection(label):
    """Extract seeds out of the raw predictions.

    :param prediction: Raw prediction of a bead image
        :type prediction:
    :return: Binarized raw prediction, bead seeds, number of beads in the image
    """

    beads = label[:, :] > 0.5
    seeds = measure.label(beads, connectivity=1, background=0)
    bead_seeds = np.zeros(shape=beads.shape, dtype=np.bool_)
    props_seeds = measure.regionprops(seeds)
    for i in range(len(props_seeds)):
        centroid = np.round(props_seeds[i].centroid).astype(np.uint16)
        bead_seeds[tuple(centroid)] = True

    beads = np.expand_dims(beads, axis=-1)
    bead_seeds = np.expand_dims(bead_seeds, axis=-1)

    num_beads = np.sum(bead_seeds)

    return beads, bead_seeds, int(num_beads)


class OriginBeadDataset(Dataset):
    """Bead data set for bead detection"""

    def __init__(self, root_dir, img_ids, transform=lambda x: x, device="cuda"):
        """

        :param root_dir: Path to the dataset containing a train, val and test directory.
            :type root_dir: pathlib.PosixPath or pathlib.WindowsPath
        :param mode: Use training, validation or test data set
            :type mode: str
        :param apply_dilation: Use dilated seeds or not.
            :type apply_dilation: bool
        :param transform: Transforms/augmentations to apply
            :type torchvision.transforms.Compose
        """

        self.root_dir = root_dir
        self.transform = transform
        self.img_ids = img_ids
        self.device = device

    def __len__(self):
        return self.img_ids + 1

    def __getitem__(self, idx):

        filename = os.path.join(self.root_dir, str(idx) + ".tif")
        labelname = os.path.join(self.root_dir, str(idx) + "_label.tif")

        img = ski.io.imread(filename)
        label = ski.io.imread(labelname)

        label = (label > 0).astype(np.uint8)
        # img = torch.from_numpy(img).float().to(self.device)
        # label = torch.from_numpy(label).float().to(self.device)
        sample = {"image": img, "label": label, "id": idx}

        sample = self.transform(sample)

        return sample


class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        sample = self.subset[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.subset)


class MaskRcnnDatasetFromSubset(Dataset):
    """Bead data set for bead detection"""

    def __init__(self, subset, transform=None, device="cuda", radius=2):
        self.subset = subset
        self.transform = transform
        self.device = device
        self.radius = radius

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        sample = self.subset[index]
        if self.transform:
            sample = self.transform(sample)
        img, label, _ = sample
        label = label[0]
        bead, bead_seeds, num_beads = seed_detection(label)
        pos = np.argwhere(bead_seeds[:, :, 0] > 0.5)
        radius = self.radius
        targets = []
        for p in pos:
            x1, y1, x2, y2 = p[0] - radius, p[1] - radius, p[0] + radius, p[1] + radius
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > label.shape[0]:
                x2 = label.shape[0]
            if y2 > label.shape[1]:
                y2 = label.shape[1]
            mask = np.zeros_like(label)
            rr, cc = ski.draw.disk(p, radius, shape=label.shape)
            mask[rr, cc] = 1
            target = {
                "boxes": torch.as_tensor([x1, y1, x2, y2], dtype=torch.float32).to(
                    self.device
                ),
                "labels": torch.as_tensor(1).to(self.device),
                "masks": torch.as_tensor(mask).to(self.device),
            }
            targets.append(target)

        # img = torch.from_numpy(img).float().to(self.device)
        # masks = torch.as_tensor(masks, dtype=torch.uint8).to(self.device)
        # normlize img to [0, 1]
        # img = (img - img.min()) / (img.max() - img.min())
        # targets = torch.as_tensor(targets)
        # sample = {"image": img, "label": label, "id": idx}
        # return img, {"boxes": boxes, "labels": mask_labels, "masks": masks}
        img = img.to(self.device)
        label = label.to(self.device)
        return img, targets, label


class MaskBeadDataset(Dataset):
    """Bead data set for bead detection"""

    def __init__(
        self,
        root_dir,
        img_ids,
        transform=lambda x: x,
        device="cuda",
        radius=3,
    ):
        """

        :param root_dir: Path to the dataset containing a train, val and test directory.
            :type root_dir: pathlib.PosixPath or pathlib.WindowsPath
        :param mode: Use training, validation or test data set
            :type mode: str
        :param apply_dilation: Use dilated seeds or not.
            :type apply_dilation: bool
        :param transform: Transforms/augmentations to apply
            :type torchvision.transforms.Compose
        """

        self.root_dir = root_dir
        self.transform = transform
        self.img_ids = img_ids
        self.radius = radius
        self.device = device

    def __len__(self):
        return self.img_ids

    def __getitem__(self, idx):

        filename = os.path.join(self.root_dir, str(idx) + ".tif")
        labelname = os.path.join(self.root_dir, str(idx) + "_label.tif")

        img = ski.io.imread(filename)
        label = ski.io.imread(labelname)
        img = ski.color.gray2rgb(img)
        img = img / 4095

        label = (label > 0).astype(np.uint8)
        pos = np.argwhere(label > 0.5)
        # pos = np.argwhere(bead_seeds[:, :, 0] > 0.5)
        radius = self.radius
        targets = []
        boxes = []
        labels = []
        masks = []
        for p in pos:
            x1, y1, x2, y2 = p[0] - radius, p[1] - radius, p[0] + radius, p[1] + radius
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > label.shape[0]:
                x2 = label.shape[0]
            if y2 > label.shape[1]:
                y2 = label.shape[1]
            mask = np.zeros_like(label)
            rr, cc = ski.draw.disk(p, self.radius, shape=label.shape)

            mask[rr, cc] = 1
            boxes.append((x1, y1, x2, y2))
            masks.append(mask)
        labels = torch.ones((len(pos),), dtype=torch.int64)
        target = {
            "labels": torch.as_tensor(labels),
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "masks": torch.as_tensor(masks),
        }
        img = img.astype(np.float32)
        img, target = self.transform(img, target)
        return img, target
