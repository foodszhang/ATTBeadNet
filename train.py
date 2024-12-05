import argparse
import math
from tqdm import tqdm, trange
import numpy as np
import json
import os
import os.path as osp
from pathlib import Path
import pandas as pd
import time

import torch
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from torch.cuda import amp

from torch.optim.lr_scheduler import ReduceLROnPlateau


from model import build_unet3plus, UNet3Plus
from torch.utils.data import DataLoader
from config.config import cfg
from utils.loss import get_loss
from utils.logger import AverageMeter, SummaryLogger
from utils.metrics import StreamSegMetrics
from utils.mytransforms import augmentors
from datasets import MaskBeadDataset, OriginBeadDataset, DatasetFromSubset
from utils import cal_score_origin, cal_maskrcnn_score
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
)
from torchvision.models.segmentation import fcn_resnet101
from model.yolo import YoloBody
from model.yolo_training import YOLOLoss
import utils.transforms as T
from utils.maskrcnn import reduce_dict
import cv2
import utils.bbox as bbox
import skimage as ski


def collate_fn(batch):
    return tuple(zip(*batch))


def base_transform():
    transforms = []
    transforms.append(T.ToTensor())

    return T.Compose(transforms)


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor

    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.NormTrans(0.4))
    transforms.append(T.ToTensor())

    return T.Compose(transforms)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


class Trainer:

    global_iter = 0
    start_epoch = 0
    epoch = 0  # current epoch
    loss_dict = dict()
    val_loss_dict = dict()
    val_f1_dict = dict()
    val_score_dict = None
    best_val_score_dict = None
    best_val_loss = 100
    best_val_f1 = 0
    val_dict = dict()

    def __init__(self, cfg, model, train_loader, val_loader):
        self.cfg_all = cfg
        # build metrics
        self.metrics = StreamSegMetrics(cfg.data.num_classes)

        cfg = self.cfg = cfg.train

        save_dir = osp.join(cfg.logger.log_dir, cfg.save_name)
        os.makedirs(save_dir, exist_ok=True)
        hyp_path = osp.join(save_dir, cfg.save_name + ".yaml")
        with open(hyp_path, "w") as f:
            f.write(cfg.dump())

        self.model: UNet3Plus = model
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader

        # build loss
        # self.criterion = build_u3p_loss(cfg.loss_type, cfg.aux_weight)
        self.criterion = get_loss("bce_dice")
        self.scaler = amp.GradScaler(
            enabled=cfg.device == "cuda"
        )  # mixed precision training

        # build optimizer
        if cfg.optimizer == "sgd":
            self.optimizer = SGD(
                self.model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                momentum=cfg.momentum,
                nesterov=cfg.nesterov,
            )
        elif cfg.optimizer == "adam":
            self.optimizer = Adam(
                self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer == "adamw":
            self.optimizer = AdamW(self.model.parameters(), lr=cfg.lr)
        else:
            raise ValueError("Unknown optimizer")
        if cfg.scheduler == "linear":
            self.lr_func = (
                lambda x: (1 - x / (cfg.epochs - 1)) * (1.0 - cfg.lrf) + cfg.lrf
            )  # linear
        elif cfg.scheduler == "cyclic":
            self.lr_func = one_cycle(1, cfg.lrf, cfg.epochs)
        else:
            raise ValueError("Unknown scheduler")

        # build scheduler
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_func)
        # self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.25, patience=13, verbose=True, min_lr=6e-5)
        self.logger = SummaryLogger(self.cfg_all)

        self.model.to(cfg.device)
        if cfg.resume:
            self.resume(cfg.resume)
        self.val_results = pd.DataFrame(columns=["epoch", "loss", "f1", "acc"])

    def resume(self, resume_path):
        print("resuming from {}".format(resume_path))
        saved = torch.load(resume_path, map_location=self.cfg.device)
        self.model.load_state_dict(saved["state_dict"])
        self.optimizer.load_state_dict(saved["optimizer"])
        self.scheduler.load_state_dict(saved["scheduler"])
        self.scheduler.step()
        self.epoch = saved["epoch"] + 1
        self.start_epoch = saved["epoch"] + 1
        self.global_iter = saved["global_iter"]

    def train(self):
        for epoch in range(self.start_epoch, self.cfg.epochs):
            self.logger.info(f"start training {epoch+1}/{self.cfg.epochs}")
            self.train_one_epoch()
            self.end_train_epoch()

    def train_one_epoch(self):
        model = self.model
        model.train()
        device = self.cfg.device
        pbar = enumerate(self.train_loader)
        num_batches = len(self.train_loader)
        # batch_size = self.train_loader.batch_size
        batch_size = 1
        accum_steps = self.cfg.accum_steps

        pbar = tqdm(
            pbar,
            total=num_batches,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b} epoch: "
            + f"{self.epoch + 1}/{self.cfg.epochs}",
        )  # progress bar
        for i, batch in pbar:
            self.warmup()
            self.global_iter += batch_size
            with amp.autocast():
                if self.cfg_all.model.name == "Unet3":
                    imgs, masks = batch[0].to(device), batch[1].to(device)
                    preds = model(imgs)["final_pred"]
                    loss = self.criterion(preds, masks)
                elif self.cfg_all.model.name == "MaskRCNN":

                    imgs, targets = batch
                    imgs = list(image.to(device) for image in imgs)
                    targets = [
                        {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in t.items()
                        }
                        for t in targets
                    ]
                    loss_dict = model(imgs, targets)
                    loss = sum(loss for loss in loss_dict.values())
                    loss_dict_reduced = reduce_dict(loss_dict)
                    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                    loss_value = losses_reduced.item()

                    if not math.isfinite(loss_value):
                        print(f"Loss is {loss_value}, stopping training")
                        print(loss_dict_reduced)
                        import sys

                        sys.exit(1)
                elif self.cfg_all.model.name == "Yolo":

                    imgs, targets = batch
                    labels = []
                    for i, t in enumerate(targets):
                        nL = len(t["boxes"])
                        labels_out = np.zeros((nL, 6))
                        box = t["boxes"]
                        box[:, [0, 2]] = box[:, [0, 2]] / 128
                        box[:, [1, 3]] = box[:, [1, 3]] / 128
                        # box[:, [0, 2]] = box[:, [0, 2]]
                        # box[:, [1, 3]] = box[:, [1, 3]]
                        box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
                        box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
                        labels_out[:, 1] = 0
                        labels_out[:, 2:] = box[:, :4]
                        labels_out[:, 0] = i
                        labels.append(labels_out)
                    labels = torch.from_numpy(np.concatenate(labels, 0)).to(device)
                    imgs = torch.stack(imgs).to(device)
                    outputs = model(imgs)
                    loss = self.criterion(outputs, labels, imgs)
                    loss = loss[0]
                    # outputs = self.bbox_utils.decode_box(outputs)
                if self.cfg_all.model.name == "resnet":
                    imgs, masks = batch[0].to(device), batch[1].to(device)
                    imgs = torch.stack(
                        [imgs[:, 0, :, :], imgs[:, 0, :, :], imgs[:, 0, :, :]], dim=1
                    ).to(device)
                    preds = model(imgs)["out"]
                    loss = self.criterion(preds, masks)

            self.update_loss_dict(self.loss_dict, {"loss": loss})
            self.scaler.scale(loss).backward()
            if (i + 1) % accum_steps == 0 or i == num_batches - 1:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        pbar.close()

    def end_train_epoch(self):
        self.epoch += 1
        if self.epoch % self.cfg.val_interval == 0 or self.epoch == self.cfg.epochs:
            val_dict = self.val_score_dict = self.validate()
            val_loss = self.val_loss_dict["loss"]
            val_f1 = self.val_f1_dict["f1"]
            train_loss = self.loss_dict["loss"]
            # if  val_loss.avg < self.best_val_loss:
            #    self.best_val_loss = val_loss.avg
            #    self.save_checkpoint(self.cfg.save_name + '_best.ckpt')

            if val_f1.avg > self.best_val_f1:
                self.best_val_f1 = val_f1.avg
                self.save_checkpoint(self.cfg.save_name + "_best.ckpt")
                data = {
                    "f1": val_f1.avg,
                    "P": self.val_dict["P"],
                    "per_time": self.val_dict["per_time"],
                    "R": self.val_dict["R"],
                }
                with open("best_f1.json", "w") as f:
                    json.dump(data, f)
            self.log_results()
        self.save_checkpoint(self.cfg.save_name + "_last.ckpt")
        self.scheduler.step()

    def save_checkpoint(self, save_name):
        state = {
            "epoch": self.epoch,
            "global_iter": self.global_iter,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(
            state, osp.join(self.cfg.logger.log_dir, self.cfg.save_name, save_name)
        )

    def warmup(self):
        ni = self.global_iter

        warmup_iters = max(self.cfg.warmup_iters, len(self.train_loader) * 3)
        if ni <= warmup_iters:
            xi = [0, warmup_iters]  # x interp
            for j, x in enumerate(self.optimizer.param_groups):
                x["lr"] = np.interp(
                    ni,
                    xi,
                    [
                        0.1 if j == 2 else 0.0,
                        x["initial_lr"] * self.lr_func(self.epoch),
                    ],
                )
                if "momentum" in x:
                    x["momentum"] = np.interp(ni, xi, [0.8, self.cfg.momentum])

    def update_loss_dict(self, loss_dict, batch_loss_dict=None):
        if batch_loss_dict is None:
            if loss_dict is None:
                return
            for k in loss_dict:
                loss_dict[k].reset()
        elif len(loss_dict) == 0:
            for k, v in batch_loss_dict.items():
                loss_dict[k] = AverageMeter(val=v)
        else:
            for k, v in batch_loss_dict.items():
                loss_dict[k].update(v)

    def log_results(self):
        log_dict = {"Train": {}, "Val": {}}

        for k, v in self.loss_dict.items():
            log_dict["Train"][k] = v.avg
        print(
            "eeeee loss:",
            self.loss_dict["loss"].avg,
            "f1:",
            self.val_f1_dict["f1"].avg,
            "P:",
            self.val_dict["P"],
            "R:",
            self.val_dict["R"],
            "label_sum:",
            self.val_dict["label_sum"],
            "pred_sum:",
            self.val_dict["pred_sum"],
        )
        self.update_loss_dict(self.loss_dict, None)
        log_dict["Train"]["lr"] = self.optimizer.param_groups[0]["lr"]

        for k, v in self.val_loss_dict.items():
            log_dict["Val"][k] = v.avg
        for k, v in self.val_f1_dict.items():
            log_dict["Val"][k] = v.avg
        self.update_loss_dict(self.val_loss_dict, None)
        self.update_loss_dict(self.val_f1_dict, None)

        # for k, v in self.val_score_dict.items():
        #    if k == "Class IoU":
        #        print(v)
        #        # self.logger.cmd_logger.info(v)
        #        continue
        #    log_dict["Val"][k] = v
        self.logger.summary(log_dict, self.global_iter)
        print("qqqq lr:", log_dict["Train"]["lr"])

    def validate(self):
        """Do validation and return specified samples"""
        self.metrics.reset()
        self.model.eval()
        device = self.cfg.device
        pbar = enumerate(self.val_loader)
        pbar = tqdm(
            pbar,
            total=len(self.val_loader),
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        )  # progress bar
        LOSS = 0
        TP, P, R, f1 = 0, 0, 0, 0
        label_sum, pred_sum = 0, 0
        with torch.no_grad():
            if self.cfg_all.model.name == "Unet3":
                for i, batch in pbar:
                    images, labels, _ = batch
                    images = images.to(device)
                    # labels = labels.to(device, dtype=torch.long)
                    labels = labels.to(device)

                    outputs = self.model(images)
                    predictions = torch.sigmoid(outputs)
                    preds = predictions.detach().cpu().numpy()
                    targets = labels.cpu().numpy()
                    targets = targets[:, 0, :, :]
                    preds = preds[:, 0, :, :]
                    for pred, target in zip(preds, targets):
                        tp, label, p = cal_score_origin(target, pred)
                        TP += tp
                        label_sum += label
                        pred_sum += p
                    # self.metrics.update(targets, preds)
                    loss = self.criterion(outputs, labels)
                    LOSS += loss
                LOSS /= len(self.val_loader)
                if pred_sum != 0:
                    P = TP / pred_sum
                else:
                    P = 0
                R = TP / label_sum
                if P + R == 0:
                    f1 = 0
                else:
                    f1 = 2 * P * R / (P + R)
                acc = R

                self.update_loss_dict(self.val_loss_dict, {"loss": LOSS})
                self.update_loss_dict(self.val_f1_dict, {"f1": f1})
                self.val_results = pd.concat(
                    [
                        self.val_results,
                        pd.DataFrame(
                            {
                                "epoch": self.epoch,
                                "loss": LOSS.detach().cpu().numpy(),
                                "f1": f1,
                                "acc": acc,
                            },
                            index=[self.epoch],
                        ),
                    ],
                    ignore_index=True,
                )
                self.val_results.to_csv("val_results.csv")

                score = self.metrics.get_results()
                pbar.close()
                return score
            elif self.cfg_all.model.name == "MaskRCNN":
                start_time = time.time()
                total_sum = 0
                for i, batch in pbar:
                    imgs, targets = batch
                    imgs = list(image.to(device) for image in imgs)
                    predictions = self.model(imgs)
                    output_dir = "./output"
                    os.makedirs(output_dir, exist_ok=True)
                    for i, prediction in enumerate(predictions):
                        img = imgs[i].mul(255).permute(1, 2, 0).byte().cpu().numpy()
                        masks = prediction["masks"]
                        ori_masks = targets[i]["masks"]
                        masks = masks.detach().cpu().numpy()
                        ori_masks = ori_masks.detach().cpu().numpy()
                        tp, label, p = cal_maskrcnn_score(ori_masks, masks)
                        TP += tp
                        label_sum += label
                        pred_sum += p
                        total_sum += 1

                if pred_sum != 0:
                    P = TP / pred_sum
                else:
                    P = 0
                R = TP / label_sum
                if P + R == 0:
                    f1 = 0
                else:
                    f1 = 2 * P * R / (P + R)
                acc = R

                self.update_loss_dict(self.val_loss_dict, {"loss": 1})

                self.update_loss_dict(self.val_f1_dict, {"f1": f1})
                self.val_dict["f1"] = f1
                self.val_dict["P"] = P
                self.val_dict["R"] = R
                self.val_dict["per_time"] = (time.time() - start_time) / total_sum
                self.val_dict["label_sum"] = label_sum
                self.val_dict["pred_sum"] = pred_sum
                # score = self.metrics.get_results()
                # print("val_loss:", LOSS)
                # print("val_f1:", f1)
            elif self.cfg_all.model.name == "Yolo":
                start_time = time.time()
                total_sum = 0
                confidence = 0.1
                num_classes = 1
                for n, batch in pbar:
                    imgs, targets = batch
                    imgs = torch.stack(imgs).to(device)
                    outputs = self.model(imgs)
                    outputs = self.bbox_util.decode_box(outputs)
                    results = self.bbox_util.non_max_suppression(
                        torch.cat(outputs, 1),
                        num_classes,
                        (128, 128),
                        (128, 128),
                        False,
                        conf_thres=confidence,
                        nms_thres=0.01,
                    )
                    for j, result in enumerate(results):
                        if result is None:
                            continue
                        img = imgs[j].mul(255).permute(1, 2, 0).byte().cpu().numpy()
                        top_label = np.array(result[:, 6], dtype="int32")
                        top_conf = result[:, 4] * result[:, 5]
                        top_boxes = result[:, :4]
                        back_img = np.zeros((128, 128, 3), dtype=np.uint8)
                        for i, c in list(enumerate(top_label)):
                            box = top_boxes[i]
                            score = top_conf[i]
                            print("66666", i, j, score)
                            top, left, bottom, right = box

                            top = max(0, np.floor(top).astype("int32"))
                            left = max(0, np.floor(left).astype("int32"))
                            bottom = min(img.shape[1], np.floor(bottom).astype("int32"))
                            right = min(img.shape[0], np.floor(right).astype("int32"))
                            rr, cc = ski.draw.rectangle(
                                (top, left), end=(bottom, right), shape=back_img.shape
                            )
                            back_img[rr, cc] = [0, 255, 0]

                        ski.io.imsave(f"./output/batch_{n}_{j}_mask.png", back_img)
                        ski.io.imsave(f"./output/batch_{n}_{j}.png", img)

                self.update_loss_dict(self.val_loss_dict, {"loss": 1})
                self.update_loss_dict(self.val_f1_dict, {"f1": f1})
                self.val_dict["f1"] = f1
                self.val_dict["P"] = P
                self.val_dict["R"] = R
                # self.val_dict["per_time"] = (time.time() - start_time) / total_sum
                self.val_dict["per_time"] = 0
                self.val_dict["label_sum"] = label_sum
                self.val_dict["pred_sum"] = pred_sum
            elif self.cfg_all.model.name == "resnet":
                for i, batch in pbar:
                    images, labels, _ = batch
                    images = images.to(device)
                    # labels = labels.to(device, dtype=torch.long)
                    labels = labels.to(device)
                    images = torch.stack(
                        [images[:, 0, :, :], images[:, 0, :, :], images[:, 0, :, :]],
                        dim=1,
                    ).to(device)

                    outputs = self.model(images)["out"]
                    predictions = torch.sigmoid(outputs)
                    preds = predictions.detach().cpu().numpy()
                    targets = labels.cpu().numpy()
                    targets = targets[:, 0, :, :]
                    preds = preds[:, 0, :, :]
                    for pred, target in zip(preds, targets):
                        tp, label, p = cal_score_origin(target, pred)
                        TP += tp
                        label_sum += label
                        pred_sum += p
                    # self.metrics.update(targets, preds)
                    loss = self.criterion(outputs, labels)
                    LOSS += loss
                LOSS /= len(self.val_loader)
                if pred_sum != 0:
                    P = TP / pred_sum
                else:
                    P = 0
                R = TP / label_sum
                if P + R == 0:
                    f1 = 0
                else:
                    f1 = 2 * P * R / (P + R)
                acc = R

                self.update_loss_dict(self.val_loss_dict, {"loss": LOSS})
                self.update_loss_dict(self.val_f1_dict, {"f1": f1})
                self.val_results = pd.concat(
                    [
                        self.val_results,
                        pd.DataFrame(
                            {
                                "epoch": self.epoch,
                                "loss": LOSS.detach().cpu().numpy(),
                                "f1": f1,
                                "acc": acc,
                            },
                            index=[self.epoch],
                        ),
                    ],
                    ignore_index=True,
                )
                self.val_dict["f1"] = f1
                self.val_dict["P"] = P
                self.val_dict["R"] = R
                # self.val_dict["per_time"] = (time.time() - start_time) / total_sum
                self.val_dict["per_time"] = 0
                self.val_dict["label_sum"] = label_sum
                self.val_dict["pred_sum"] = pred_sum

                score = self.metrics.get_results()
                pbar.close()
                return score


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

    import torch
    import random
    import numpy as np

    seed = cfg.train.seed
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model, data = cfg.model, cfg.data
    if model.name == "Unet3":
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
        data_transforms = augmentors(augmentation="train", min_value=0, max_value=4095)
        dataset = OriginBeadDataset(root_dir=Path("./data/"), img_ids=1000)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [800, 200])
        train_dataset = DatasetFromSubset(
            train_dataset, transform=data_transforms["train"]
        )
        val_dataset = DatasetFromSubset(val_dataset, transform=data_transforms["val"])

        train_loader = DataLoader(
            train_dataset,
            batch_size=data.batch_size,
            shuffle=True,
            num_workers=data.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=data.batch_size,
            shuffle=False,
            num_workers=data.num_workers,
        )

        trainer = Trainer(cfg, model, train_loader, val_loader)
        trainer.train()
    elif model.name == "MaskRCNN":
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

        train_dataset = MaskBeadDataset(
            root_dir=Path("./data/20240911_60X_flat_clip/"),
            img_ids=130,
            transform=get_transform(train=True),
        )
        val_dataset = MaskBeadDataset(
            root_dir=Path("./data/20240911_60X_flat_clip_test/"),
            img_ids=18,
            transform=get_transform(train=False),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=data.batch_size,
            shuffle=True,
            num_workers=data.num_workers,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=data.batch_size,
            shuffle=False,
            num_workers=data.num_workers,
            collate_fn=collate_fn,
        )

        trainer = Trainer(cfg, model, train_loader, val_loader)
        trainer.train()
    elif model.name == "Yolo":
        input_shape = [128, 128]
        phi = "l"
        # anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        #
        anchors_mask = [[2], [1], [0]]
        num_classes = 1
        model = YoloBody(anchors_mask=anchors_mask, num_classes=num_classes, phi=phi)
        # anchors = [
        #    [10, 13, 16, 30, 33, 23],
        #    [30, 61, 62, 45, 59, 119],
        #    [116, 90, 156, 198, 373, 326],
        # ]
        anchors = [[4.5594, 4.9717], [5.5252, 4.8209], [6.0084, 6.0117]]
        anchors = np.array(anchors).reshape(-1, 2)
        yolo_loss = YOLOLoss(anchors, num_classes, input_shape, anchors_mask, 0)

        train_dataset = MaskBeadDataset(
            root_dir=Path("./data/20240911_60X_flat_clip/"),
            img_ids=130,
            transform=get_transform(train=True),
        )
        val_dataset = MaskBeadDataset(
            root_dir=Path("./data/20240911_60X_flat_clip_test/"),
            img_ids=18,
            transform=get_transform(train=False),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=data.batch_size,
            shuffle=True,
            num_workers=data.num_workers,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=data.batch_size,
            shuffle=False,
            num_workers=data.num_workers,
            collate_fn=collate_fn,
        )

        trainer = Trainer(cfg, model, train_loader, val_loader)
        trainer.criterion = yolo_loss
        trainer.bbox_util = bbox.DecodeBox(
            anchors, num_classes, input_shape, anchors_mask
        )
        trainer.train()
    elif model.name == "resnet":
        model = fcn_resnet101(num_classes=1)
        data_transforms = augmentors(augmentation="train", min_value=0, max_value=4095)
        dataset = OriginBeadDataset(
            root_dir=Path("./data/20240911_60X_flat_multi"), img_ids=130
        )
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [110, 21])
        train_dataset = DatasetFromSubset(
            train_dataset, transform=data_transforms["train"]
        )
        val_dataset = DatasetFromSubset(val_dataset, transform=data_transforms["val"])

        train_loader = DataLoader(
            train_dataset,
            batch_size=data.batch_size,
            shuffle=True,
            num_workers=data.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=data.batch_size,
            shuffle=False,
            num_workers=data.num_workers,
        )

        trainer = Trainer(cfg, model, train_loader, val_loader)
        trainer.train()


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
