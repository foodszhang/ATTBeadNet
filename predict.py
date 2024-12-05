from model import build_unet3plus, UNet3Plus
from config.config import cfg
import torch
import argparse
import skimage as ski
import utils
from utils.mytransforms import min_max_normalization
import matplotlib.pyplot as plt
from skimage import measure
from skimage import data, restoration, util
import numpy as np
import os
import skimage as ski
from utils import cal_score_origin, get_center
import shutil
import time

from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)


def main(args):
    cfg.merge_from_file(args.cfg)
    if args.seed is not None:
        cfg.train.seed = int(args.seed)
    if args.resume:
        cfg.train.resume = args.resume
    if args.data_dir:
        cfg.data.data_dir = args.data_dir
    if args.use_tensorboard is not None:
        cfg.train.logger.use_tensorboard = args.use_tensorboard == 1
    elif args.use_wandb is not None:
        cfg.train.logger.use_wandb = args.use_wandb == 1
    cfg.freeze()

    import random

    seed = 42
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = "cpu"
    model = None
    if cfg.model.name == "Unet3":
        model, data = cfg.model, cfg.data
        model = build_unet3plus(
            data.num_classes,
            model.encoder,
            model.skip_ch,
            model.aux_losses,
            model.use_cgm,
            model.pretrained,
            model.dropout,
            am="CBAM",
        )
    elif cfg.model.name == "MaskRCNN":
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

        model = maskrcnn_resnet50_fpn()
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        num_classes = 2
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
    # checkpoint = torch.load("./runs/u3p_bead/u3p_bead_last.ckpt")
    checkpoint = torch.load("./runs/u3p_bead/u3p_bead_best.ckpt")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    base_dir = "./data/fig1/"
    # base_dir = "./data/20240911_60X_flat_clip_test/"
    data_dir = f"{base_dir}"
    result_dir = "./results/20241203_fig1/"
    os.makedirs(result_dir, exist_ok=True)
    sum = 0
    for imgname in os.listdir(data_dir):
        if (
            not imgname.endswith(".tif")
            or imgname.endswith("label.tif")
            or imgname.endswith("mask.tif")
        ):
            continue
        img = ski.io.imread(os.path.join(data_dir, imgname))
        pure_img = img.copy()
        pure_img = ski.exposure.rescale_intensity(pure_img, out_range=(0, 255))
        pure_img = ski.color.gray2rgb(pure_img)
        img = np.expand_dims(img, axis=-1)
        img, pads, status = utils.zero_pad_model_input(img_upsampled=img)
        net_input = min_max_normalization(img=img, min_value=0, max_value=4095)
        net_input = np.transpose(np.expand_dims(net_input, axis=0), [0, 3, 1, 2])
        net_input = torch.from_numpy(net_input)
        if cfg.model.name == "Unet3":
            outputs = model(net_input)
            target = outputs
            prediction = torch.sigmoid(target)
            p = prediction[0][0]
            p = p.detach().cpu().numpy()
            _, bead_seeds, num_beads = seed_detection(prediction)
            bead_seeds = bead_seeds[pads[0] :, pads[1] :, :]
            pos = np.argwhere(bead_seeds[:, :, 0] > 0.5)

            for p in pos:
                rr, cc = ski.draw.circle_perimeter(*p, 2, shape=pure_img.shape)
                pure_img[rr, cc] = [255, 0, 0]
            ski.io.imsave(
                f'{result_dir}{imgname.split(".")[0]}.png', pure_img.astype(np.uint8)
            )
            ski.io.imsave(f'{result_dir}{imgname.split(".")[0]}_mask.tif', bead_seeds)
        elif cfg.model.name == "MaskRCNN":
            img = ski.io.imread(os.path.join(data_dir, imgname))
            img = np.expand_dims(img, axis=-1)
            img, pads, status = utils.zero_pad_model_input(img_upsampled=img)
            img = ski.color.gray2rgb(img[:, :, 0])
            img = img / 4095.0
            predict_base = np.zeros((img.shape[0], img.shape[1]))
            for row in range(0, img.shape[0], 128):
                for col in range(0, img.shape[1], 128):
                    net_input = img[row : row + 128, col : col + 128]
                    # net_input = min_max_normalization(img=img, min_value=0, max_value=4095)
                    net_input = np.transpose(
                        np.expand_dims(net_input, axis=0), [0, 3, 1, 2]
                    )
                    net_input = torch.from_numpy(net_input)
                    # print("123123123", net_input[:, :, 128:256, 128:256].shape)
                    # net_input = net_input[:, :, :128, :128]
                    net_input = net_input.type(torch.float32)

                    outputs = model(net_input)

                    for i, prediction in enumerate(outputs):
                        p_masks = prediction["masks"]
                        if not isinstance(p_masks, torch.Tensor):
                            continue
                        predict_poses = [get_center(mask[0]) for mask in p_masks]
                        predict_poses = [
                            pose for pose in predict_poses if pose is not None
                        ]
                        print("123123123qqq", predict_poses)
                        for pose in predict_poses:
                            pose = (pose[0] + row, pose[1] + col)
                            rr, cc = ski.draw.disk(pose, 3, shape=predict_base.shape)
                            predict_base[rr, cc] = 1

            _, pred_bead_seed, _ = seed_detection(predict_base)
            bead_seeds = pred_bead_seed[pads[0] :, pads[1] :, :]
            pos = np.argwhere(bead_seeds[:, :, 0] > 0.5)
            print("123123123", pos)
            sum += pos.shape[0]
            for p in pos:
                rr, cc = ski.draw.circle_perimeter(*p, 4, shape=pure_img.shape)
                pure_img[rr, cc] = [255, 0, 0]
            ski.io.imsave(
                f'{result_dir}{imgname.split(".")[0]}.png',
                pure_img.astype(np.uint8),
            )
            ski.io.imsave(f'{result_dir}{imgname.split(".")[0]}_mask.tif', bead_seeds)
        elif cfg.model.name == "resnet":
            img = ski.io.imread(os.path.join(data_dir, imgname))
            img = ski.color.gray2rgb(img)
            img, pads, status = utils.zero_pad_model_input(img_upsampled=img)
            net_input = min_max_normalization(img=img, min_value=0, max_value=4095)
            net_input = np.transpose(np.expand_dims(net_input, axis=0), [0, 3, 1, 2])
            net_input = torch.from_numpy(net_input)
            outputs = model(net_input)
            target = outputs
            prediction = torch.sigmoid(target)
            p = prediction[0][0]
            p = p.detach().cpu().numpy()
            _, bead_seeds, num_beads = seed_detection(prediction)
            bead_seeds = bead_seeds[pads[0] :, pads[1] :, :]
            pos = np.argwhere(bead_seeds[:, :, 0] > 0.5)

            for p in pos:
                rr, cc = ski.draw.circle_perimeter(*p, 2, shape=pure_img.shape)
                pure_img[rr, cc] = [255, 0, 0]
            ski.io.imsave(
                f'{result_dir}{imgname.split(".")[0]}.png', pure_img.astype(np.uint8)
            )
            ski.io.imsave(f'{result_dir}{imgname.split(".")[0]}_mask.tif', bead_seeds)
    print("123123123", sum)


def seed_detection(prediction):
    """Extract seeds out of the raw predictions.

    :param prediction: Raw prediction of a bead image
        :type prediction:
    :return: Binarized raw prediction, bead seeds, number of beads in the image
    """

    # beads = prediction[0, 0, :, :] > 0.5
    beads = prediction[:, :] > 0.5
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation network")

    parser.add_argument(
        "--cfg",
        help="experiment configure file name",
        default="config/bead.yaml",
        type=str,
    )
    parser.add_argument("--seed", help="random seed", default=None)
    parser.add_argument(
        "--resume", help="resume from checkpoint", default=None, type=str
    )
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--use_wandb", default=None, type=int)
    parser.add_argument("--use_tensorboard", default=None, type=int)

    args = parser.parse_args()
    main(args)
