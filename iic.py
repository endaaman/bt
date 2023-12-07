import os
import sys
import math
import json
import re
from glob import glob
import itertools

import cv2
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import transforms
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics as skmetrics
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pydantic import Field
# from timm.scheduler.cosine_lr import CosineLRScheduler

import pytorch_grad_cam as CAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget, ClassifierOutputTarget

from endaaman import with_wrote, load_images_from_dir_or_file, grid_split, with_mkdir
from endaaman.ml import BaseTrainerConfig, BaseTrainer, Checkpoint, pil_to_tensor
from endaaman.ml.metrics import MultiAccuracy
from endaaman.ml.cli import BaseMLCLI, BaseDLArgs, BaseTrainArgs

from models import IICModel, CrossEntropyLoss
from datasets.fold import IICFoldDataset


np.set_printoptions(suppress=True, floatmode='fixed')
J = os.path.join


def compute_joint(x_out, x_tf_out):
    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)

    p_i_j = (p_i_j + p_i_j.t()) / 2.

    p_i_j = p_i_j / p_i_j.sum()

    return p_i_j


def iic_loss(x_out, x_tf_out, EPS=sys.float_info.epsilon):
    bs, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    p_i = torch.clip(p_i, min=EPS)
    p_j = torch.clip(p_j, min=EPS)
    p_i_j = torch.clip(p_i_j, min=EPS)
    # p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
    # p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    # p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    alpha = 2.0

    loss = torch.log(p_i_j) - alpha * torch.log(p_j) - alpha * torch.log(p_i)
    loss = -1 * (p_i_j * loss).sum()

    return loss

class TrainerConfig(BaseTrainerConfig):
    model_name:str
    source: str
    fold: int
    total_fold: int
    num_classes: int
    size: int
    minimum_area: float
    limit: int = -1
    upsample: bool = False

    num_classes: int = 10
    num_classes_over: int = 100


class Trainer(BaseTrainer):
    def prepare(self):
        self.criterion = CrossEntropyLoss(input_logits=True)
        self.fig_col_count = 2
        return IICModel(
            name=self.config.model_name,
            num_classes=self.config.num_classes,
            num_classes_over=self.config.num_classes_over
        )

    def eval(self, i0, i1):
        p0, p0_over = self.model(i0.to(self.device), activate=True)
        p1, p1_over = self.model(i1.to(self.device), activate=True)

        loss = iic_loss(p0, p1)
        loss_over = iic_loss(p0_over, p1_over)
        total_loss = loss + loss_over

        # total_loss = 1/-total_loss
        return total_loss, None

    def create_scheduler(self):
        return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)

    def continues(self):
        lr = self.get_current_lr()
        return lr > 1e-6

    def get_metrics(self):
        return {
            # 'acc': MultiAccuracy(),
        }

class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        pass

    class TrainArgs(BaseTrainArgs):
        lr: float = 0.01
        batch_size: int = Field(16, cli=('--batch-size', '-B', ))
        num_workers: int = Field(10, cli=('--num-workers', '-N' ))
        num_classes: int = Field(10, cli=('--num-classes', ))
        num_classes_over: int = Field(100, cli=('--num-classes-over', ))
        minimum_area: float = 0.7
        limit: int = -1
        upsample: bool = Field(False, cli=('--upsample', ))
        epoch: int = Field(100, cli=('--epoch', '-E'))
        total_fold: int = Field(5, cli=('--total-fold', ))
        fold: int = 0
        model_name: str = Field('efficientnet_b0', cli=('--model', '-m'))
        source: str = Field('enda3_512', cli=('--source', ))
        suffix: str = ''
        prefix: str = ''
        size: int = Field(512, cli=('--size', '-s'))
        overwrite: bool = Field(False, cli=('--overwrite', '-o'))

    def run_train(self, a:TrainArgs):
        config = TrainerConfig(
            model_name = a.model_name,
            batch_size = a.batch_size,
            lr = a.lr,
            source = 'enda3_512',
            size = a.size,
            total_fold = a.total_fold,
            fold = a.fold,
            minimum_area = a.minimum_area,
            limit = a.limit,
            upsample = a.upsample,

            num_classes = a.num_classes,
            num_classes_over = a.num_classes_over,
        )

        dss = [
            IICFoldDataset(
                 total_fold=a.total_fold,
                 fold=a.fold,
                 source_dir=J('cache', a.source),
                 target=t,
                 code='LMGAOB',
                 size=a.size,
                 minimum_area=a.minimum_area,
                 limit=a.limit,
                 upsample = a.upsample,
                 image_aug=True,
                 aug_mode='same',
                 normalize=True,
            ) for t in ('train', 'test')
        ]

        out_dir = J(
            'out', 'iic',
            f'fold{a.total_fold}_{a.fold}', a.prefix, config.model_name
        )
        if a.suffix:
            out_dir += f'_{a.suffix}'

        trainer = Trainer(
            config=config,
            out_dir=out_dir,
            train_dataset=dss[0],
            val_dataset=dss[1],
            use_gpu=not a.cpu,
            overwrite=a.overwrite,
            experiment_name=a.source,
        )

        trainer.start(a.epoch)



if __name__ == '__main__':
    cli = CLI()
    cli.run()
