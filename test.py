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
from utils import cal_score_origin
import shutil
import pandas as pd
from utils.loss import get_loss
import time


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
    print(cfg)

    import random

    seed = 42
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = "cpu"
    model, data = cfg.model, cfg.data
    model_name = model.name
    if model_name == "Unet3":
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

    elif model_name == "MaskRCNN":
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
    # checkpoint = torch.load('./results/test/20X/u3p_bead_best_with_cbam.ckpt')
    checkpoint = torch.load("./runs/u3p_bead/u3p_bead_best.ckpt")
    # checkpoint = torch.load('../dl_dataset/20240911_60X_flat_clip_test/u3p_bead_best.ckpt')
    # checkpoint = torch.load('./results/20240911rec/u3p_bead_best.ckpt')
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model = model.to(device)
    # img = ski.io.imread('../001_img_upsampled.tif')
    # data_dir = '../dl_dataset/20240614/800fm/40X/BF/'
    # data_dir = '../dl_dataset/20240911_20X_flat_clip_test/'
    # data_dir = '../dl_dataset/test-more/20X/test/'
    # data_dir = '../dl_dataset/20240807_new/0.1 mgml hrp/bg/'
    data_dir = "../dl_dataset/20241011/"
    # data_dir = '../dl_dataset/flu/60X'
    # os.makedirs('./results/test/0X', exist_ok=True)
    result_dir = "./results/20241011/60X/"
    os.makedirs(result_dir, exist_ok=True)
    # shutil.copy('./runs/u3p_bead/u3p_bead_best.ckpt', '../dl_dataset/20240911_20X_flat_clip_test/u3p_bead_best.ckpt')
    # shutil.copy('./runs/u3p_bead/u3p_bead_best.ckpt', './results/paper_img/60X/u3p_bead_best_with_cbam.ckpt')
    pred_sum = 0
    tp_sum = 0
    label_sum = 0
    bead_sum = 0
    file_num = 0
    total_time = 0
    start_time = time.time()
    for imgname in os.listdir(data_dir):
        if not imgname.endswith(".tif") or imgname.endswith("label.tif"):
            continue
        file_start_time = time.time()
        # imgname = '800fm_10.tif'
        # labelname = '800fm_10_mask.tif'
        # imgname = '3.tif'
        img = ski.io.imread(os.path.join(data_dir, imgname))
        label_name = os.path.join(data_dir, imgname.split(".")[0] + "_label.tif")
        pure_img = img.copy()
        pure_img = ski.exposure.rescale_intensity(pure_img, out_range=(0, 255))
        pure_img = ski.color.gray2rgb(pure_img)
        label_img = ski.io.imread(label_name)
        label_padding = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=-1)
        img, pads, status = utils.zero_pad_model_input(img_upsampled=img)
        label_padding, _, _ = utils.zero_pad_model_input(img_upsampled=label_padding)
        # net_input = min_max_normalization(img=img, min_value=0, max_value=4095)
        net_input = min_max_normalization(img=img, min_value=0, max_value=255)
        net_input = np.transpose(np.expand_dims(net_input, axis=0), [0, 3, 1, 2])
        label_padding = np.transpose(
            np.expand_dims(label_padding, axis=0), [0, 3, 1, 2]
        )
        net_input = torch.from_numpy(net_input)
        net_input = net_input.to(device)
        # prediction = torch.sigmoid(model(net_input))
        cost_time = time.time()
        outputs = model(net_input)
        if model_name == "Unet3":
            print("cost time:", time.time() - cost_time)
            prediction = torch.sigmoid(outputs)
            p = prediction[0][0]
            target = outputs
            p = p[pads[0] :, pads[1] :]
            tp, label_num, pred_num = cal_score_origin(label_img, p)
            tp_sum += tp
            label_sum += label_num
            pred_sum += pred_num
            _, bead_seeds, num_beads = seed_detection(prediction)
            bead_seeds = bead_seeds[pads[0] :, pads[1] :, :]
            pos = np.argwhere(bead_seeds[:, :, 0] > 0.5)
            bead_sum += num_beads
            # ski.io.imsave(f'{result_dir}predict_{imgname}', p)
            for p in pos:
                rr, cc = ski.draw.circle_perimeter(*p, 4, shape=pure_img.shape)
                pure_img[rr, cc] = [255, 0, 0]
            ski.io.imsave(
                f'{result_dir}{imgname.split(".")[0]}.png', pure_img.astype(np.uint8)
            )
            # shutil.copy(label_name, result_dir)
            file_num += 1
            total_time += time.time() - file_start_time
            print("end time:", time.time() - file_start_time, imgname)
        elif model_name == "MaskRCNN":
            for i, prediction in enumerate(outputs):
                masks = prediction["masks"]
                masks = masks.detach().cpu().numpy()
                ori_masks = ori_masks.detach().cpu().numpy()
                tp, label, p = cal_maskrcnn_score(ori_masks, masks)
                TP += tp
                label_sum += label
                pred_sum += p

    P = tp_sum / pred_sum
    R = tp_sum / label_sum
    per_time = total_time / file_num
    f1 = 2 * P * R / (P + R)
    acc = R

    pd.DataFrame(
        {"P": [P], "R": [R], "f1": [f1], "num": [bead_sum], "label_num": [label_sum]}
    ).to_csv(f"{result_dir}result.csv", header=False, index=False)
    print(
        f"P: {P}, R:{R}, f1: {f1}, num: {pred_sum}, label_num: {label_sum}, per_time:{per_time}"
    )
    # print('!!!!!!', np.max(p), np.min(p), np.mean(p))
    # pos = np.argwhere(p > 0.5)
    ## draw circle with pos
    # fig = plt.gcf()
    # ax = fig.gca()
    # for x, y in pos:
    #    c = plt.Circle((y, x), 3, color='yellow', linewidth=2, fill=False)
    #    ax.add_patch(c)

    # plt.tight_layout()
    # model = UNet_3Plus_DeepSup()


def seed_detection(prediction):
    """Extract seeds out of the raw predictions.

    :param prediction: Raw prediction of a bead image
        :type prediction:
    :return: Binarized raw prediction, bead seeds, number of beads in the image
    """

    beads = prediction[0, 0, :, :] > 0.5
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
