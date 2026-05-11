"""Training script for K-class bead center heatmap prediction using MultiSizeBeadTileDataset."""

import argparse
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from config.config import cfg
from datasets import MultiSizeBeadTileDataset
from model import build_unet3plus
from utils.loss import get_loss
from utils.metrics import (
    extract_gt_centers_from_mask,
    extract_pred_centers_from_prob,
    hungarian_match,
)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    """One-cycle learning rate scheduler function."""
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def build_class_balanced_sampler(dataset, cfg):
    """Build a WeightedRandomSampler that oversamples tiles with rare classes."""
    df = dataset.df
    weights = []
    stats = {"negative": 0, "class_0_only": 0, "class_1_only": 0, "mixed": 0}

    for _, row in df.iterrows():
        counts = []
        for k in range(dataset.num_classes):
            counts.append(row.get(f"count_class_{k}", 0))

        total = sum(counts)

        if total == 0:
            w = cfg.data.sampler_weights.negative
            stats["negative"] += 1
        else:
            active_classes = [k for k, c in enumerate(counts) if c > 0]

            if dataset.num_classes == 2:
                c0, c1 = counts[0], counts[1]

                if c0 > 0 and c1 > 0:
                    w = cfg.data.sampler_weights.mixed
                    stats["mixed"] += 1
                elif c1 > 0:
                    w = cfg.data.sampler_weights.class_1_only
                    stats["class_1_only"] += 1
                elif c0 > 0:
                    w = cfg.data.sampler_weights.class_0_only
                    stats["class_0_only"] += 1
                else:
                    w = cfg.data.sampler_weights.negative
                    stats["negative"] += 1
            else:
                # K-class fallback: weight based on rarity of class composition
                active_set = set(active_classes)
                # Give higher weight to tiles containing rarer classes (higher index)
                max_class = max(active_set) if active_set else 0
                if max_class == 0:
                    w = cfg.data.sampler_weights.class_0_only
                    stats["class_0_only"] += 1
                elif len(active_set) == 1:
                    # Single non-zero class
                    if max_class == 1:
                        w = cfg.data.sampler_weights.class_1_only
                        stats["class_1_only"] += 1
                    else:
                        # Classes > 1 treated as rare (use class_1_only weight as proxy)
                        w = cfg.data.sampler_weights.class_1_only
                        stats["class_1_only"] += 1
                else:
                    # Multiple classes present
                    w = cfg.data.sampler_weights.mixed
                    stats["mixed"] += 1

        weights.append(float(w))

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )

    return sampler, stats


class Trainer:
    """Trainer for MultiSizeBeadTileDataset with UNet3Plus model."""

    def __init__(self, cfg, model, train_loader, val_loader):
        self.cfg_all = cfg
        cfg = self.cfg = cfg.train

        # Create save directory
        save_dir = os.path.join(cfg.log_dir, cfg.save_name)
        os.makedirs(save_dir, exist_ok=True)

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss and mixed precision
        self.criterion = get_loss(cfg.loss_type)
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.device == "cuda")

        # Optimizer
        if cfg.optimizer == "adamw":
            self.optimizer = AdamW(
                self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

        # Scheduler (one-cycle)
        if cfg.scheduler == "cyclic":
            self.lr_func = one_cycle(1, cfg.lrf, cfg.epochs)
        elif cfg.scheduler == "linear":
            self.lr_func = lambda x: (1 - x / (cfg.epochs - 1)) * (1.0 - cfg.lrf) + cfg.lrf
        else:
            raise ValueError(f"Unknown scheduler: {cfg.scheduler}")

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_func)

        # Training state
        self.epoch = 0
        self.best_f1 = 0.0
        self.global_iter = 0

    def train_one_epoch(self):
        """Train for one epoch."""
        self.model.train()
        device = self.cfg.device

        pbar = tqdm(
            self.train_loader,
            desc=f"Training epoch {self.epoch + 1}/{self.cfg.epochs}",
        )

        for batch in pbar:
            images = batch["image"].float().to(device)
            masks = batch["mask"].float().to(device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.cfg.device == "cuda"):
                logits = self.model(images)["final_pred"]
                loss = self.criterion(logits, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.global_iter += images.shape[0]
            pbar.set_postfix({"loss": loss.item()})

        self.optimizer.zero_grad()
        pbar.close()

    def validate(self):
        """Compute per-class precision/recall/F1 using center-based Hungarian matching."""
        self.model.eval()
        K = self.cfg_all.data.num_classes
        thresholds = self.cfg_all.postprocess.thresholds
        min_distances = self.cfg_all.postprocess.min_distances
        match_radius = self.cfg_all.postprocess.match_radius
        class_names = self.cfg_all.data.class_names

        tp = [0.0] * K
        fp = [0.0] * K
        fn = [0.0] * K
        count_err = [0.0] * K
        n_samples = [0] * K

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch["image"].float().to(self.cfg.device)
                masks = batch["mask"].float().cpu().numpy()
                counts = batch["count"].float().cpu().numpy()
                # Use center_mask for GT extraction, not heatmap
                center_masks = batch.get("center_mask", masks)

                out = self.model(images)
                logits = out["final_pred"] if isinstance(out, dict) else out
                probs = torch.sigmoid(logits).cpu().numpy()

                B = images.shape[0]
                for b in range(B):
                    gt_centers = extract_gt_centers_from_mask(center_masks[b], mode="pixel")
                    pred_centers = extract_pred_centers_from_prob(
                        probs[b], thresholds, min_distances, use_peak_local_max=True
                    )

                    matched, unmatched_pred, unmatched_gt = hungarian_match(
                        pred_centers, gt_centers, match_radius
                    )

                    for k in range(K):
                        pred_k = [p for p in pred_centers if p["class_id"] == k]
                        gt_k = [g for g in gt_centers if g["class_id"] == k]
                        matched_k = [m for m in matched if pred_centers[m[0]]["class_id"] == k]

                        tp[k] += len(matched_k)
                        fp[k] += len(pred_k) - len(matched_k)
                        fn[k] += len(gt_k) - len(matched_k)

                        pred_count = len(pred_k)
                        gt_count = int(counts[b, k])
                        count_err[k] += abs(pred_count - gt_count)
                        n_samples[k] += 1

        rows = []
        f1s = []
        for k in range(K):
            p = tp[k] / (tp[k] + fp[k]) if (tp[k] + fp[k]) > 0 else 0.0
            r = tp[k] / (tp[k] + fn[k]) if (tp[k] + fn[k]) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            mae = count_err[k] / n_samples[k] if n_samples[k] > 0 else 0.0
            rows.append({
                "class_id": k,
                "class_name": class_names[k] if k < len(class_names) else str(k),
                "precision": p,
                "recall": r,
                "f1": f1,
                "mae_count": mae,
            })
            f1s.append(f1)

        macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

        df = pd.DataFrame(rows)
        df.to_csv("val_metrics.csv", index=False)

        return {"macro_f1": macro_f1, "per_class": rows}

    def fit(self):
        """Run the full training loop."""
        for epoch in range(self.cfg.epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.cfg.epochs}")

            self.train_one_epoch()

            if (epoch + 1) % self.cfg.val_interval == 0:
                metrics = self.validate()
                print(f"macro_f1: {metrics['macro_f1']:.4f}")
                for row in metrics["per_class"]:
                    print(
                        f"  class {row['class_name']}: "
                        f"P={row['precision']:.3f} R={row['recall']:.3f} "
                        f"F1={row['f1']:.4f} MAE={row['mae_count']:.2f}"
                    )

                # Save best checkpoint
                if metrics["macro_f1"] > self.best_f1:
                    self.best_f1 = metrics["macro_f1"]
                    self.save_checkpoint(f"{self.cfg.save_name}_best.ckpt")
                    print(f"  Saved best checkpoint (macro_f1={self.best_f1:.4f})")

            self.save_checkpoint(f"{self.cfg.save_name}_last.ckpt")
            self.scheduler.step()

    def save_checkpoint(self, name):
        """Save model checkpoint."""
        save_path = os.path.join(self.cfg.log_dir, self.cfg.save_name, name)
        torch.save(
            {
                "epoch": self.epoch,
                "global_iter": self.global_iter,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "best_f1": self.best_f1,
            },
            save_path,
        )


def main(args):
    """Main entry point."""
    cfg.merge_from_file(args.cfg)
    cfg.freeze()

    # Set seeds
    seed = cfg.train.seed
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Data
    print(f"Training target mode: {cfg.target.mode}, radius_per_class: {cfg.target.radius_per_class}")
    train_ds = MultiSizeBeadTileDataset(
        root_dir=cfg.data.train_root,
        split="train_pool",
        target_mode=cfg.target.mode,
        radius_per_class=cfg.target.radius_per_class,
    )
    val_ds = MultiSizeBeadTileDataset(
        root_dir=cfg.data.train_root,
        split="internal_val",
        target_mode=cfg.target.mode,
        radius_per_class=cfg.target.radius_per_class,
    )

    num_train = len(train_ds)
    print(f"num train tiles: {num_train}")

    if cfg.data.sampler == "class_balanced":
        sampler, stats = build_class_balanced_sampler(train_ds, cfg)
        print(f"num negative tiles: {stats['negative']}")
        print(f"num class_0_only tiles: {stats['class_0_only']}")
        print(f"num class_1_only tiles: {stats['class_1_only']}")
        print(f"num mixed tiles: {stats['mixed']}")
        print(f"sampler mode: class_balanced")
        print(f"weights: negative={cfg.data.sampler_weights.negative}, "
              f"class_0_only={cfg.data.sampler_weights.class_0_only}, "
              f"class_1_only={cfg.data.sampler_weights.class_1_only}, "
              f"mixed={cfg.data.sampler_weights.mixed}")
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.data.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=cfg.data.num_workers,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    # Model
    model = build_unet3plus(
        num_classes=cfg.data.num_classes,
        encoder=cfg.model.encoder,
        skip_ch=cfg.model.skip_ch,
        aux_losses=cfg.model.aux_losses,
        use_cgm=cfg.model.use_cgm,
        pretrained=cfg.model.pretrained,
        dropout=cfg.model.dropout,
    ).to(cfg.train.device)

    trainer = Trainer(cfg, model, train_loader, val_loader)
    trainer.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MultiSize K-class bead model")
    parser.add_argument(
        "--cfg",
        help="Path to config file",
        default="config/multisize_bead.yaml",
        type=str,
    )
    args = parser.parse_args()
    main(args)
