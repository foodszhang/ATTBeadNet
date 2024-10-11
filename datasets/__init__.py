from torch.utils.data import DataLoader , Dataset
from PIL import Image
import torchvision.transforms.functional as F
import skimage as ski
from pathlib import Path
import numpy as np
from typing import Tuple
import os

class OriginBeadDataset(Dataset):
    """ Bead data set for bead detection """

    def __init__(self, root_dir,  img_ids, transform=lambda x: x):
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

    def __len__(self):
        return self.img_ids + 1

    def __getitem__(self, idx):


        filename = os.path.join(self.root_dir, str(idx)+'.tif')
        labelname = os.path.join(self.root_dir, str(idx)+ '_label.tif')

        img = ski.io.imread(filename)
        label = ski.io.imread(labelname)

        label = (label > 0).astype(np.uint8)
        sample = {'image': img, 'label': label, 'id': idx}

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
