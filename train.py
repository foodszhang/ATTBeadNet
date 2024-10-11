import argparse
import math
from tqdm import tqdm
import numpy as np
import os
import os.path as osp
from pathlib import Path
import pandas as pd

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
from datasets import OriginBeadDataset, DatasetFromSubset
from utils import cal_score_origin

def one_cycle(y1=0.0, y2=1.0, steps=100):
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

class Trainer:

    global_iter = 0
    start_epoch = 0
    epoch = 0   # current epoch
    loss_dict = dict()
    val_loss_dict = dict()
    val_f1_dict = dict()
    val_score_dict = None
    best_val_score_dict = None
    best_val_loss = 100
    best_val_f1 = 0
    
    def __init__(self, cfg, model, train_loader, val_loader):
        self.cfg_all = cfg

        # build metrics
        self.metrics = StreamSegMetrics(cfg.data.num_classes)

        cfg = self.cfg = cfg.train

        save_dir = osp.join(cfg.logger.log_dir, cfg.save_name)
        os.makedirs(save_dir, exist_ok=True)
        hyp_path = osp.join(save_dir, cfg.save_name+'.yaml')
        with open(hyp_path, "w") as f: 
            f.write(cfg.dump())
        
        self.model: UNet3Plus = model
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader

        # build loss
        #self.criterion = build_u3p_loss(cfg.loss_type, cfg.aux_weight)
        self.criterion = get_loss('bce_dice')
        self.scaler = amp.GradScaler(enabled=cfg.device == 'cuda')  # mixed precision training

        # build optimizer
        if cfg.optimizer == 'sgd':
            self.optimizer = SGD(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum, nesterov=cfg.nesterov)
        elif cfg.optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == 'adamw':
            self.optimizer = AdamW(self.model.parameters(), lr=cfg.lr)
        else:
            raise ValueError('Unknown optimizer')
        if cfg.scheduler == 'linear':
            self.lr_func = lambda x: (1 - x / (cfg.epochs - 1)) * (1.0 - cfg.lrf) + cfg.lrf  # linear
        elif cfg.scheduler == 'cyclic':
            self.lr_func = one_cycle(1, cfg.lrf, cfg.epochs)
        else:
            raise ValueError('Unknown scheduler')

        # build scheduler
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_func)
        #self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.25, patience=13, verbose=True, min_lr=6e-5)
        self.logger = SummaryLogger(self.cfg_all)

        self.model.to(cfg.device)
        if cfg.resume:
            self.resume(cfg.resume)
        self.val_results = pd.DataFrame(columns=['epoch', 'loss', 'f1', 'acc'])
        
        
    def resume(self, resume_path):
        print('resuming from {}'.format(resume_path))
        saved = torch.load(resume_path, map_location=self.cfg.device)
        self.model.load_state_dict(saved['state_dict'])
        self.optimizer.load_state_dict(saved['optimizer'])
        self.scheduler.load_state_dict(saved['scheduler'])
        self.scheduler.step()
        self.epoch = saved['epoch'] + 1
        self.start_epoch = saved['epoch'] + 1
        self.global_iter = saved['global_iter']

    def train(self):
        for epoch in range(self.start_epoch, self.cfg.epochs):
            self.logger.info(f'start training {epoch+1}/{self.cfg.epochs}')
            self.train_one_epoch()
            self.end_train_epoch()

    def train_one_epoch(self):
        model = self.model
        model.train()
        device = self.cfg.device
        pbar = enumerate(self.train_loader)
        num_batches = len(self.train_loader)
        batch_size = self.train_loader.batch_size
        accum_steps = self.cfg.accum_steps
        
        pbar = tqdm(pbar, total=num_batches, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b} epoch: ' \
            + f'{self.epoch + 1}/{self.cfg.epochs}')  # progress bar
        for i, batch in pbar:
            self.warmup()
            #imgs, masks = batch[0].to(device), batch[1].to(device, dtype=torch.long)
            imgs, masks = batch[0].to(device), batch[1].to(device)
            self.global_iter += batch_size
            with amp.autocast():
                preds = model(imgs)['final_pred']
                loss = self.criterion(preds, masks)
            self.update_loss_dict(self.loss_dict, {'loss': loss})
            self.scaler.scale(loss).backward()
            if (i+1) % accum_steps == 0 or i == num_batches - 1:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        pbar.close()

    def end_train_epoch(self):
        self.epoch += 1
        if self.epoch % self.cfg.val_interval == 0 or self.epoch == self.cfg.epochs:
            val_dict = self.val_score_dict = self.validate()
            val_loss = self.val_loss_dict['loss']
            val_f1 = self.val_f1_dict['f1']
            train_loss = self.loss_dict['loss']
            #if  val_loss.avg < self.best_val_loss:
            #    self.best_val_loss = val_loss.avg
            #    self.save_checkpoint(self.cfg.save_name + '_best.ckpt')

            if  val_f1.avg > self.best_val_f1:
                self.best_val_f1 = val_f1.avg
                self.save_checkpoint(self.cfg.save_name + '_best.ckpt')
            self.log_results()
        self.save_checkpoint(self.cfg.save_name + '_last.ckpt')
        self.scheduler.step()
    
    def save_checkpoint(self, save_name):
        state = {
            'epoch': self.epoch,
            'global_iter': self.global_iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        torch.save(state, osp.join(self.cfg.logger.log_dir, self.cfg.save_name, save_name))

    def warmup(self):
        ni = self.global_iter

        warmup_iters = max(self.cfg.warmup_iters, len(self.train_loader.dataset) * 3)
        if ni <= warmup_iters:
            xi = [0, warmup_iters]  # x interp
            for j, x in enumerate(self.optimizer.param_groups):
                x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * self.lr_func(self.epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [0.8, self.cfg.momentum])

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
        log_dict = {
            'Train': {},
            'Val': {}
        }

        for k, v in self.loss_dict.items():
            log_dict['Train'][k] = v.avg
        self.update_loss_dict(self.loss_dict, None)
        log_dict['Train']['lr'] = self.optimizer.param_groups[0]['lr']

        for k, v in self.val_loss_dict.items():
            log_dict['Val'][k] = v.avg
        for k, v in self.val_f1_dict.items():
            log_dict['Val'][k] = v.avg
        self.update_loss_dict(self.val_loss_dict, None)
        self.update_loss_dict(self.val_f1_dict, None)
        
        for k, v in self.val_score_dict.items():
            if k == 'Class IoU':
                print(v)
                # self.logger.cmd_logger.info(v)
                continue
            log_dict['Val'][k] = v
        self.logger.summary(log_dict, self.global_iter)


    def validate(self):
        """Do validation and return specified samples"""
        self.metrics.reset()
        self.model.eval()
        device = self.cfg.device
        pbar = enumerate(self.val_loader)
        pbar = tqdm(pbar, total=len(self.val_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        LOSS = 0
        TP, P, R, f1 = 0, 0, 0, 0
        label_sum, pred_sum = 0, 0
        with torch.no_grad():

            for i, batch in pbar:
                images, labels, _ = batch
                images = images.to(device)
                #labels = labels.to(device, dtype=torch.long)
                labels = labels.to(device)

                outputs = self.model(images)
                predictions =  torch.sigmoid(outputs) 
                preds = predictions.detach().cpu().numpy()
                targets = labels.cpu().numpy()
                targets = targets[:, 0, :, :]
                preds = preds[:, 0, :, :]
                for (pred, target) in zip(preds, targets):
                    tp, label, p = cal_score_origin(target, pred)
                    TP += tp
                    label_sum += label
                    pred_sum += p
                #self.metrics.update(targets, preds)
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

            self.update_loss_dict(self.val_loss_dict, {'loss': LOSS})
            self.update_loss_dict(self.val_f1_dict, {'f1': f1})
            self.val_results = pd.concat([self.val_results, pd.DataFrame({'epoch': self.epoch, 'loss': LOSS.detach().cpu().numpy(), 'f1': f1, 'acc': acc}, index=[self.epoch])], ignore_index=True)
            self.val_results.to_csv('val_results.csv')

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
    model = build_unet3plus(data.num_classes, model.encoder, model.skip_ch, model.aux_losses, model.use_cgm, model.pretrained, model.dropout, am="CBAM")
    data_transforms = augmentors(augmentation='train', min_value=0, max_value=4095)
    dataset = OriginBeadDataset(root_dir=Path('./data/'), img_ids=1000)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [800, 200])
    train_dataset = DatasetFromSubset(train_dataset, transform=data_transforms['train'])
    val_dataset = DatasetFromSubset(val_dataset, transform=data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=data.batch_size, shuffle=True, num_workers=data.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=data.batch_size, shuffle=False, num_workers=data.num_workers)
    
    trainer = Trainer(cfg, model, train_loader, val_loader)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="config/bead.yaml",
                        type=str)
    parser.add_argument('--seed',
                        help='random seed',
                        default=None)
    parser.add_argument('--resume',
                        help='resume from checkpoint',
                        default=None,
                        type=str)
    parser.add_argument('--data_dir',
                        default=None,
                        type=str)
    parser.add_argument('--use_wandb',
                        default=None,
                        type=int)
    parser.add_argument('--use_tensorboard',
                        default=None,
                        type=int)

    args = parser.parse_args()
    main(args)
