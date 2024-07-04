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
from datasets import IICFoldDataset



torch.multiprocessing.set_sharing_strategy('file_system')
np.set_printoptions(suppress=True, floatmode='fixed')
J = os.path.join

DS_BASE = 'data/tiles'
# DS_BASE = 'cache'

class IICLoss(nn.Module):
    def __init__(self, alpha, beta=0.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, p, q):
        EPS = sys.float_info.epsilon
        bs, k = p.size()
        J = p.unsqueeze(2) * q.unsqueeze(1)
        J = J.sum(dim=0)
        J = (J + J.t()) / 2.0 / J.sum()

        J = torch.clamp(J, min=EPS)

        r = (p + q) / 2.0
        Pm = r.mean(dim=0).view(k, 1).expand(k, k)
        Qm = Pm.t()

        Ps = r.std(dim=0).view(k, 1).expand(k, k)
        Qs = Ps.t()

        a = self.alpha
        b = self.beta
        loss = J * (a*Pm.log() + a*Qm.log() - J.log() - b*Ps.log() - b*Qs.log())
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
    beta: float
    mean: float = 0.7
    std: float = 0.2

    def unique_code(self):
        return to_unique_code(self.code)


class Trainer(BaseTrainer):
    def prepare(self):
        self.criterion = IICLoss(alpha=self.config.alpha, beta=self.config.beta)
        self.fig_col_count = 2
        return IICModel(
            name=self.config.model_name,
            num_classes=self.config.num_classes,
            num_classes_over=self.config.num_classes_over
        )

    def select_inputs_and_gts(self, params):
        return None, params[2]

    def eval(self, x, y, gt):
        p, p_over = self.model(x.to(self.device), activate=True)
        q, q_over = self.model(y.to(self.device), activate=True)
        loss = self.criterion(p, q)
        loss_over = self.criterion(p_over, q_over)
        total_loss = loss + loss_over
        return total_loss, torch.argmax(p, dim=1)

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
        lr: float = -1
        base_lr: float = 1e-5 # ViT for 1e-6
        batch_size: int = Field(100, s='-B')
        code: str = 'LMGGG_'
        num_classes_over: int = 32
        minimum_area: float = 0.6
        limit: int = 500
        noupsample: bool = False
        epoch: int = Field(50, l='--epoch', s='-E')
        total_fold: int = 5
        fold: int = 0
        model_name: str = Field('resnet34', l='--model', s='-m')
        source: str = 'enda4_256'
        suffix: str = ''
        prefix: str = ''
        alpha: float = 2.0
        beta: float = 0.1
        size: int = Field(256, s='-s')
        overwrite: bool = Field(False, s='-O')

    def run_train(self, a:TrainArgs):
        lr = a.lr if a.lr>0 else a.base_lr*a.batch_size

        config = TrainerConfig(
            model_name = a.model_name,
            batch_size = a.batch_size,
            code = a.code,
            lr = lr,
            source = a.source,
            size = a.size,
            total_fold = a.total_fold,
            fold = a.fold,
            minimum_area = a.minimum_area,
            limit = a.limit,
            upsample = not a.noupsample,
            alpha = a.alpha,
            beta = a.beta,

            num_classes = len(to_unique_code(a.code)),
            num_classes_over = a.num_classes_over,
        )

        dss = [IICFoldDataset(
            total_fold=a.total_fold,
            fold=a.fold,
            source_dir=J(DS_BASE, a.source),
            target=t,
            code=config.code,
            size=a.size,
            minimum_area=a.minimum_area,
            limit=a.limit,
            upsample=config.upsample,
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
