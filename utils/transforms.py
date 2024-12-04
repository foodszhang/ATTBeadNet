from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import random
import torch
import torchvision
from torch import nn, Tensor
from torchvision import ops
from torchvision.transforms import (
    functional as F,
    InterpolationMode,
    transforms as T,
)
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapOnImage


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class NormTrans:
    def __init__(self, p):
        self.p = p
        self.seq = iaa.SomeOf(
            2,
            [
                iaa.Affine(rotate=(-90, 90)),
                iaa.Affine(scale=(0.85, 1.25)),
                iaa.CropAndPad(percent=(-0.25, 0.25)),
            ],
        )

    def __call__(self, img, target):
        if random.random() > self.p:
            return img, target
        boxes = target["boxes"].detach().cpu().numpy()
        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=bb[0], y1=bb[1], x2=bb[2], y2=bb[3]) for bb in boxes],
            shape=img.shape,
        )
        mask = target["masks"].detach().cpu().numpy()
        mask = mask.transpose(1, 2, 0)
        segmap = SegmentationMapOnImage(mask, shape=mask.shape)

        img, bbs, segmap = self.seq(
            image=img, bounding_boxes=bbs, segmentation_maps=segmap
        )
        target = {
            "boxes": torch.as_tensor(
                [[bb.x1, bb.y1, bb.x2, bb.y2] for bb in bbs.bounding_boxes]
            ),
            "masks": torch.as_tensor(segmap.get_arr().transpose(2, 0, 1)),
            "labels": target["labels"],
        }
        # aug_masks = []
        # for mask in masks:
        #    segmap = SegmentationMapOnImage(mask, shape=mask.shape)
        #    segmap = self.seq(segmentation_maps=segmap)
        #    mask = segmap.get_arr()
        #    aug_masks.append(mask)
        # img, bbs = self.seq(image=img, bounding_boxes=bbs)
        # target = {
        #    "boxes": torch.as_tensor(
        #        [[bb.x1, bb.y1, bb.x2, bb.y2] for bb in bbs.bounding_boxes]
        #    ),
        #    "masks": torch.as_tensor(aug_masks),
        #    "labels": target["labels"],
        # }
        return img, target
        return img, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(nn.Module):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                _, _, width = F.get_dimensions(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    target["masks"] = target["masks"].flip(-1)
                if "keypoints" in target:
                    keypoints = target["keypoints"]
                    keypoints = _flip_coco_person_keypoints(keypoints, width)
                    target["keypoints"] = keypoints
        return image, target


class ScaleJitter(nn.Module):
    """Randomly resizes the image and its bounding boxes  within the specified scale range.
    The class implements the Scale Jitter augmentation as described in the paper
    `"Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation" <https://arxiv.org/abs/2012.07177>`_.

    Args:
        target_size (tuple of ints): The target size for the transform provided in (height, weight) format.
        scale_range (tuple of ints): scaling factor interval, e.g (a, b), then scale is randomly sampled from the
            range a <= scale <= b.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
    """

    def __init__(
        self,
        target_size: Tuple[int, int],
        scale_range: Tuple[float, float] = (0.1, 2.0),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias=True,
    ):
        super().__init__()
        self.target_size = target_size
        self.scale_range = scale_range
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(
                    f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions."
                )
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        _, orig_height, orig_width = F.get_dimensions(image)

        scale = self.scale_range[0] + torch.rand(1) * (
            self.scale_range[1] - self.scale_range[0]
        )
        r = (
            min(self.target_size[1] / orig_height, self.target_size[0] / orig_width)
            * scale
        )
        new_width = int(orig_width * r)
        new_height = int(orig_height * r)

        image = F.resize(
            image,
            [new_height, new_width],
            interpolation=self.interpolation,
            antialias=self.antialias,
        )

        if target is not None:
            target["boxes"][:, 0::2] *= new_width / orig_width
            target["boxes"][:, 1::2] *= new_height / orig_height
            if "masks" in target:
                target["masks"] = F.resize(
                    target["masks"],
                    [new_height, new_width],
                    interpolation=InterpolationMode.NEAREST,
                    antialias=self.antialias,
                )

        return image, target
