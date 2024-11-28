import numpy as np
import skimage as ski
import scipy as sci
from skimage import measure


def zero_pad_model_input(img_upsampled):
    """Zero-pad model input to get for the model needed sizes.

    :param img_upsampled:
        :type img_upsampled:
    :param gui:
        :type gui:
    :return: zero-padded img, [0s padded in y-direction, 0s padded in x-direction], exit status
    """

    pads = []

    for i in range(0, 2):  # 0: y-pads, 1: x-pads

        if img_upsampled.shape[i] < 128:
            # Zero-pad to 128
            pads.append(128 - img_upsampled.shape[i])

        elif img_upsampled.shape[i] == 128:
            # No zero-padding needed
            pads.append(0)

        elif 128 < img_upsampled.shape[i] < 256:
            # Zero-pad to 256
            pads.append(256 - img_upsampled.shape[i])

        elif img_upsampled.shape[i] == 256:
            # No zero-padding needed
            pads.append(0)

        elif 256 < img_upsampled.shape[i] < 512:
            # Zero-pad to 512
            pads.append(512 - img_upsampled.shape[i])

        elif img_upsampled.shape[i] == 512:
            # No zero-padding needed
            pads.append(0)

        elif 512 < img_upsampled.shape[i] < 768:
            # Zero-pad to 768
            pads.append(768 - img_upsampled.shape[i])

        elif img_upsampled.shape[i] == 768:
            # No zero-padding needed
            pads.append(0)

        elif 768 < img_upsampled.shape[i] < 1024:
            # Zero-pad to 1024
            pads.append(1024 - img_upsampled.shape[i])

        elif img_upsampled.shape[i] == 1024:
            # No zero-padding needed
            pads.append(0)

        elif 1024 < img_upsampled.shape[i] < 1360:
            # Zero-pad to 1280
            pads.append(1360 - img_upsampled.shape[i])

        elif img_upsampled.shape[i] == 1360:
            # No zero-padding needed
            pads.append(0)

        elif 1360 < img_upsampled.shape[i] < 1680:
            # Zero-pad to 1680
            pads.append(1680 - img_upsampled.shape[i])

        elif img_upsampled.shape[i] == 1680:
            # No zero-padding needed
            pads.append(0)

        elif 1680 < img_upsampled.shape[i] < 2048:
            # Zero-pad to 2048
            pads.append(2048 - img_upsampled.shape[i])

        elif img_upsampled.shape[i] == 2048:
            # No zero-padding needed
            pads.append(0)

        elif 2048 < img_upsampled.shape[i] < 2560:
            # Zero-pad to 2560
            pads.append(2560 - img_upsampled.shape[i])

        elif img_upsampled.shape[i] == 2560:
            # No zero-padding needed
            pads.append(0)

        elif 2560 < img_upsampled.shape[i] < 4096:
            # Zero-pad to 4096
            pads.append(4096 - img_upsampled.shape[i])

        elif img_upsampled.shape[i] == 4096:
            # No zero-padding needed
            pads.append(0)

        else:
            return 1, 1, 1
    img_upsampled = np.pad(
        img_upsampled, ((pads[0], 0), (pads[1], 0), (0, 0)), mode="constant"
    )

    return img_upsampled, pads, 0


def cal_score_origin(label, pred):
    _, label_beed_seed, _ = seed_detection(label)
    _, pred_beed_seed, _ = seed_detection(pred)
    label_pos = np.argwhere(label_beed_seed > 0)
    pred_pos = np.argwhere(pred_beed_seed > 0)
    cost_matrix = sci.spatial.distance.cdist(label_pos, pred_pos, "euclidean")
    row, col = sci.optimize.linear_sum_assignment(cost_matrix)
    TP, TN, FP, FN = 0, 0, 0, 0
    for x, y in zip(row, col):
        if cost_matrix[x, y] < 3:
            TP += 1
    return TP, len(label_pos), len(pred_pos)


def seed_detection(prediction):
    beads = prediction > 0.5
    seeds = measure.label(beads, connectivity=1, background=0)
    bead_seeds = np.zeros(shape=beads.shape, dtype=np.bool_)
    props_seeds = measure.regionprops(seeds)
    for i in range(len(props_seeds)):
        centroid = np.round(props_seeds[i].centroid).astype(np.uint16)
        bead_seeds[tuple(centroid)] = True

    # beads = np.expand_dims(beads, axis=-1)
    # bead_seeds = np.expand_dims(bead_seeds, axis=-1)

    num_beads = np.sum(bead_seeds)

    return beads, bead_seeds, int(num_beads)
