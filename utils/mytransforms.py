import cv2
import numpy as np
import random
import scipy
import torch
import torchvision.transforms as transforms
from imgaug import augmenters as iaa


def min_max_normalization(img, min_value=None, max_value=None):
    """Min-max-normalization for images.

    :param img: Image with shape  [height, width, color channels].
        :type img:
    :param min_value: Minimum value for the normalization. All values below this value are clipped
        :type min_value: int
    :param max_value: Maximum value for the normalization. All values above this value are clipped.
        :type max_value: int
    :return: Normalized image (float32)
    """

    if max_value is None:

        max_value = img.max()

    if min_value is None:  # Get new minimum value

        min_value = img.min()

    # Clip image to filter hot and cold pixels
    img = np.clip(img, min_value, max_value)

    # Apply Min-max-normalization
    # img = 2 * (img.astype(np.float32) - min_value) / (max_value - min_value) - 1
    # img = 2 * (img.astype(np.float32) - min_value) / (max_value - min_value) - 1
    img = 2 * (img.astype(np.float32) - min_value) / (max_value - min_value) - 1

    return img.astype(np.float32)


def augmentors(augmentation, min_value, max_value):
    """Get augmentations/transforms for the training/evaluation process.

    :param augmentation: 'train' or 'eval'.
        :type augmentation: str
    :param min_value: Minimum value for the normalization. All values below this value are clipped
        :type min_value: int
    :param max_value: Maximum value for the normalization. All values above this value are clipped.
        :type max_value: int
    :return Dictionary containing the augmentations/transform for the training/evaluation process.
    """

    if augmentation == "train":
        data_transforms = {
            "train": transforms.Compose(
                [
                    Flip(p=0.75),
                    Scaling(p=0.3),
                    Rotate(p=0.3),
                    Contrast(p=0.3),
                    Blur(p=0.3),
                    Noise(p=0.3),
                    ToTensor(min_value=min_value, max_value=max_value),
                ]
            ),
            "val": ToTensor(min_value=min_value, max_value=max_value),
        }
        # data_transforms = {'train':ToTensor(min_value=min_value, max_value=max_value),
        #                   'val':ToTensor(min_value=min_value, max_value=max_value)}

    elif augmentation == "eval":
        data_transforms = ToTensorEval(min_value=min_value, max_value=max_value)

    else:
        raise Exception("Unknown transformation: {}".format(augmentation))

    return data_transforms


class Blur(object):
    """Blur augmentation (label-preserving transformation)"""

    def __init__(self, p=1):
        """

        :param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p

    def __call__(self, sample):
        """

        :param sample: Dictionary containing an image and the corresponding label image (numpy arrays), and file id.
            :type sample: dict
        :return: Dictionary containing the augmented image, label image, and file id.
        """

        img, label, img_id = sample["image"], sample["label"], sample["id"]

        if random.random() < self.p:

            sigma = 3 * random.random()
            img = scipy.ndimage.gaussian_filter(img, sigma, order=0)

        return {"image": img, "label": label, "id": img_id}


class Contrast(object):
    """Contrast augmentation (label-preserving transformation)"""

    def __init__(self, p=1):
        """

        param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p

    def __call__(self, sample):
        """

        :param sample: Dictionary containing an image and the corresponding label image (numpy arrays), and file id.
            :type sample: dict
        :return: Dictionary containing the augmented image, label image, and file id.
        """

        img, label, img_id = sample["image"], sample["label"], sample["id"]

        if random.random() < self.p:

            if random.randint(0, 1) == 0:  # CLAHE

                img = img.astype(np.float32) / 65535 * 255
                clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
                if len(img.shape) == 2:
                    img = np.expand_dims(clahe.apply(img.astype(np.uint8)), axis=-1)
                else:
                    img = np.expand_dims(
                        clahe.apply(img[:, :, 0].astype(np.uint8)), axis=-1
                    )
                img = img.astype(np.float32) / 255 * 65535
                img = img.astype(np.uint16)

            else:  # Contrast and gamma adjustment

                dtype = img.dtype
                img = (img.astype(np.float32) - np.iinfo(dtype).min) / (
                    np.iinfo(dtype).max - np.iinfo(dtype).min
                )
                contrast_range, gamma_range = (0.65, 1.35), (0.5, 1.5)

                # Contrast
                img_mean, img_min, img_max = img.mean(), img.min(), img.max()
                factor = np.random.uniform(contrast_range[0], contrast_range[1])
                img = (img - img_mean) * factor + img_mean

                # Gamma
                img_mean, img_std, img_min, img_max = (
                    img.mean(),
                    img.std(),
                    img.min(),
                    img.max(),
                )
                gamma = np.random.uniform(gamma_range[0], gamma_range[1])
                rnge = img_max - img_min
                img = (
                    np.power(((img - img_min) / float(rnge + 1e-7)), gamma) * rnge
                    + img_min
                )
                if random.random() < 0.5:
                    img = img - img.mean() + img_mean
                    img = img / (img.std() + 1e-8) * img_std

                img = np.clip(img, 0, 1)
                img = (
                    img * (np.iinfo(dtype).max - np.iinfo(dtype).min)
                    - np.iinfo(dtype).min
                )
                img = img.astype(dtype)

        return {"image": img, "label": label, "id": img_id}


class Flip(object):
    """Flip and rotation augmentation (label-preserving transformation)"""

    def __init__(self, p=0.5):
        """

        param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p

    def __call__(self, sample):
        """

        :param sample: Dictionary containing an image and the corresponding label image (numpy arrays), and file id.
            :type sample: dict
        :return: Dictionary containing the augmented image, label image, and file id.
        """
        img, label, img_id = sample["image"], sample["label"], sample["id"]

        if random.random() < self.p:

            # img.shape: (imgWidth, imgHeight, imgChannels)
            if img.shape[0] == img.shape[1]:
                h = random.randint(0, 2)
            else:
                h = random.randint(0, 1)

            if h == 0:  # Flip left-right

                img = np.flip(img, axis=1)
                label = np.flip(label, axis=1)

            elif h == 1:  # Flip up-down

                img = np.flip(img, axis=0)
                label = np.flip(label, axis=0)

            elif h == 2:  # Rotate 90°

                img = np.rot90(img, axes=(0, 1))
                label = np.rot90(label, axes=(0, 1))

        return {"image": img.copy(), "label": label.copy(), "id": img_id}


class Noise(object):
    """Gaussian noise augmentation"""

    def __init__(self, p=0.25):
        """

        param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p

    def __call__(self, sample):
        """

        :param sample: Dictionary containing an image and the corresponding label image (numpy arrays), and file id.
            :type sample: dict
        :return: Dictionary containing the augmented image, label image, and file id.
        """

        img, label, img_id = sample["image"], sample["label"], sample["id"]

        if random.random() < self.p:

            # Add noise with sigma 1-7% of image maximum
            sigma = random.randint(1, 7) / 100 * np.max(img)

            # Add noise to selected images
            seq = iaa.Sequential(
                [
                    iaa.AdditiveGaussianNoise(
                        scale=sigma, per_channel=False, deterministic=False
                    )
                ]
            )
            img = seq.augment_image(img)

        return {"image": img, "label": label, "id": img_id}


class Rotate(object):
    """Rotation augmentation (label-changing augmentation)"""

    def __init__(self, p=1):
        """

        param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p

    def __call__(self, sample):
        """

        :param sample: Dictionary containing an image and the corresponding label image (numpy arrays), and file id.
            :type sample: dict
        :return: Dictionary containing the augmented image, label image, and file id.
        """

        img, label, img_id = sample["image"], sample["label"], sample["id"]

        angle = (-180, 180)

        if random.random() < self.p:
            angle = random.uniform(angle[0], angle[1])

            seq1 = iaa.Sequential([iaa.Affine(rotate=angle, deterministic=True)])
            seq2 = iaa.Sequential(
                [iaa.Affine(rotate=angle, deterministic=True, order=0)]
            )
            img = seq1.augment_image(img)
            label = seq2.augment_image(label)

        return {"image": img, "label": label, "id": img_id}


class Scaling(object):
    """Scaling augmentation (label-changing transformation)"""

    def __init__(self, p=1):
        """

        param p: Probability to apply augmentation to an image.
            :type p: float
        """
        self.p = p

    def __call__(self, sample):
        """

        :param sample: Dictionary containing an image and the corresponding label image (numpy arrays), and file id.
            :type sample: dict
        :return: Dictionary containing the augmented image, label image, and file id.
        """

        img, label, img_id = sample["image"], sample["label"], sample["id"]

        scale = (0.65, 1.35)

        if random.random() < self.p:
            scale1 = random.uniform(scale[0], scale[1])
            scale2 = random.uniform(scale[0], scale[1])
            seq1 = iaa.Sequential([iaa.Affine(scale={"x": scale1, "y": scale2})])
            seq2 = iaa.Sequential(
                [iaa.Affine(scale={"x": scale1, "y": scale2}, order=0)]
            )
            img = seq1.augment_image(img)
            label = seq2.augment_image(label)

        return {"image": img.copy(), "label": label.copy(), "id": img_id}


class ToTensor(object):
    """Convert image and label image to Torch tensors"""

    def __init__(self, min_value, max_value):

        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, sample):
        """

        :param sample: Dictionary containing an image and the corresponding label image (numpy arrays), and file id.
            :type sample: dict
        :return: Image and label image (torch tensors) and file id (str).
        """

        img, label, img_id = sample["image"], sample["label"], sample["id"]

        # Normalize image
        img = min_max_normalization(
            img, min_value=self.min_value, max_value=self.max_value
        )

        # Swap axes from (H, W, Channels) to (Channels, H, W)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        if len(label.shape) == 2:
            label = np.expand_dims(label, axis=-1)
        img = np.transpose(img, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)

        return img.to(torch.float), label.to(torch.float), img_id


class ToTensorEval(object):
    """Convert image and label image to Torch tensors"""

    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, sample):
        """

        :param sample: Dictionary containing an image and the corresponding label image (numpy arrays), and file id.
            :type sample: dict
        :return: Image and label image (torch tensors) and file id (str).
        """

        img, label, img_id = sample["image"], sample["label"], sample["id"]

        # Normalize image
        img = min_max_normalization(
            img, min_value=self.min_value, max_value=self.max_value
        )

        # Swap axes from (H, W, Channels) to (Channels, H, W)
        img = np.transpose(img, (2, 0, 1))

        img = torch.from_numpy(img)

        # Due to int32 bug in pytorch
        label = torch.from_numpy(label.astype(int))

        return img.to(torch.float), label, img_id
