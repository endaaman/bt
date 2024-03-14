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

from endaaman import with_wrote,  with_mkdir
from endaaman.image import load_images_from_dir_or_file, grid_split
from endaaman.ml import BaseTrainerConfig, BaseTrainer, Checkpoint, pil_to_tensor
from endaaman.ml.metrics import MultiAccuracy
from endaaman.ml.cli import BaseMLCLI, BaseDLArgs, BaseTrainArgs

from models import IICModel, CrossEntropyLoss
from datasets.fold import IICFoldDataset


np.set_printoptions(suppress=True, floatmode='fixed')
J = os.path.join


class IICLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha)

    def forward(self, i, j):
        # EPS = sys.float_info.epsilon
        EPS = torch.finfo(torch.float32).eps
        bs, k = i.size()
        p_i_j = i.unsqueeze(2) * j.unsqueeze(1)
        p_i_j = p_i_j.sum(dim=0)
        # p_i_j = (p_i_j + p_i_j.t()) / 2.0
        # p_i_j = torch.clamp(p_i_j, min=EPS)
        p_i_j += p_i_j.clone().t()
        p_i_j /= 2.0
        p_i_j /= p_i_j.sum()
        p_i_j.clamp_(min=EPS)

        # p_i = p_i_j.sum(dim=0).view(1, k).expand(k, k)
        # p_j = p_i.t()
        p_i = p_i_j.sum(dim=0).view(1, k).expand(k, k)
        p_j = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
        # p_j = p_i.t()
        # p_j = p_i_j.sum(dim=0).view(1, k).repeat(1, 4)
        # p_i = p_i_j.sum(dim=0).view(k, 1).repeat(1, k)

        loss = p_i_j * (self.alpha*torch.log(p_j) + self.alpha*torch.log(p_i) - torch.log(p_i_j))
        loss = loss.sum()
        return loss

def to_unique_code(code):
    return [c for c in dict.fromkeys(code) if c in 'LMGAOB']


class TrainerConfig(BaseTrainerConfig):
    model_name:str
    source: str
    fold: int
    total_fold: int
    code: str
    num_classes: int
    num_classes_over: int
    size: int
    minimum_area: float
    limit: int
    upsample: bool

    alpha: float
    mean: float = 0.7
    std: float = 0.2

    def unique_code(self):
        return to_unique_code(self.code)


class Trainer(BaseTrainer):
    def prepare(self):
        self.criterion = IICLoss(alpha=self.config.alpha)
        self.fig_col_count = 2
        return IICModel(
            name=self.config.model_name,
            num_classes=self.config.num_classes,
            num_classes_over=self.config.num_classes_over
        )

    def select_inputs_and_gts(self, params):
        return params[0], params[2]

    def eval(self, x, y, gt):
        # torch.cuda.empty_cache()
        x, x_over = self.model(x.to(self.device), activate=True)
        y, y_over = self.model(y.to(self.device), activate=True)
        loss = self.criterion(x, y)
        loss_over = self.criterion(x_over, y_over)
        total_loss = loss + loss_over
        return total_loss, torch.argmax(x, dim=1)

    def metrics_precision(self, preds, gts, batch):
        preds = preds.detach().cpu()
        labels = torch.unique(preds)
        correct = 0
        for label in labels:
            items = gts[preds == label]
            elements, counts = torch.unique(items, return_counts=True)
            dominant = elements[torch.argmax(counts)]
            # print(label, dominant, torch.sum(items == dominant))
            correct += torch.sum(items == dominant)
        return correct/len(preds)


class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        pass

    class TrainArgs(BaseTrainArgs):
        lr: float = 0.001
        batch_size: int = Field(128, s='-B')
        code: str = 'LMGGG_'
        num_classes_over: int = 30
        minimum_area: float = 0.6
        limit: int = 1000
        upsample: bool = False
        epoch: int = Field(50, l='--epoch', s='-E')
        total_fold: int = 5
        fold: int = 0
        model_name: str = Field('resnet34', l='--model', s='-m')
        source: str = 'enda4_256'
        suffix: str = ''
        prefix: str = ''
        alpha: float = 2.0
        size: int = Field(256, s='-s')
        overwrite: bool = Field(False, s='-O')

    def run_train(self, a:TrainArgs):
        config = TrainerConfig(
            model_name = a.model_name,
            batch_size = a.batch_size,
            code = a.code,
            lr = a.lr,
            source = a.source,
            size = a.size,
            total_fold = a.total_fold,
            fold = a.fold,
            minimum_area = a.minimum_area,
            limit = a.limit,
            upsample = a.upsample,
            alpha = a.alpha,

            num_classes = len(to_unique_code(a.code)),
            num_classes_over = a.num_classes_over,
        )

        dss = [IICFoldDataset(
            total_fold=a.total_fold,
            fold=a.fold,
            source_dir=J('data/tiles', a.source),
            target=t,
            code=config.code,
            size=a.size,
            minimum_area=a.minimum_area,
            limit=a.limit,
            upsample = a.upsample,
            augmentation=True,
            normalization=True,
            mean=config.mean,
            std=config.std,
        ) for t in ('train', 'test')]

        out_dir = J(
            'out', 'iic', a.source,
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
