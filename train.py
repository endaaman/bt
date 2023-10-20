import os
import json
import re
from glob import glob

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
# import pytorch_grad_cam as CAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from endaaman.ml import BaseMLCLI, BaseDLArgs, BaseTrainerConfig, BaseTrainer
from datasets.fold import FoldDataset


J = os.path.join

class TrainerConfig(BaseTrainerConfig):
    model_name:str
    source: str
    fold: int
    total_fold: int
    code: str
    num_classes: int
    size: int
    minimum_area: float


class Trainer(BaseTrainer):
    def prepare(self):
        return TimmModel(name=self.config.model_name, num_classes=num_classes)

    def eval(self, inputs, gts):
        preds = self.model(inputs.to(self.device), activate=False)
        return loss, preds.detach().cpu()

    def _visualize_confusion(self, ax, label, preds, gts):
        preds = torch.argmax(preds, dim=-1)
        gts = gts.flatten()
        cm = skmetrics.confusion_matrix(gts.numpy(), preds.numpy())
        ticks = [*self.train_dataset.unique_code]
        sns.heatmap(cm, annot=True, ax=ax, fmt='g', xticklabels=ticks, yticklabels=ticks)
        ax.set_title(label)
        ax.set_xlabel('Predict', fontsize=13)
        ax.set_ylabel('GT', fontsize=13)

    def visualize_train_confusion(self, ax, train_preds, train_gts, val_preds, val_gts):
        self._visualize_confusion(ax, 'train', train_preds, train_gts)

    def visualize_val_confusion(self, ax, train_preds, train_gts, val_preds, val_gts):
        self._visualize_confusion(ax, 'val', val_preds, val_gts)

    def get_metrics(self):
        return {
            'acc': MultiAccuracy(),
        }

class CLI(BaseMLCLI):
    class CommonArgs(BaseDLArgs):
        pass

    class TrainArgs(CommonArgs):
        lr: float = 0.0002
        batch_size: int = Field(16, cli=('--batch-size', '-B', ))
        num_workers: int = 4
        minimum_area: float = 0.7
        epoch: int = 50
        total_fold: int = Field(6, cli=('--total-fold', ))
        fold: int = 0
        model_name: str = Field('tf_efficientnet_b0', cli=('--model', '-m'))
        source: str = Field('images_enda2_512', cli=('--source', ))
        suffix: str = ''
        size: int = Field(512, cli=('--size', '-s'))
        code: str = 'LMGAO'
        experiment_name:str = Field('Default', cli=('--exp', ))
        overwrite: bool = Field(False, cli=('--overwrite', '-o'))

    def run_train(self, a:TrainArgs):
        num_classes = len(set([*a.code]) - {'_'})

        config = TrainerConfig(
            model_name = a.model_name,
            batch_size = a.batch_size,
            lr = a.lr,
            source = a.source,
            size = a.size,
            code = a.code,
            total_fold = a.total_fold,
            fold = a.fold,
            num_classes = num_classes,
            minimum_area = a.minimum_area,
        )

        source_dir = J('cache/', a.source)

        dss = [
            FoldDataset(
                 total_fold=a.total_fold,
                 fold=a.fold,
                 source_dir=source_dir,
                 target=t,
                 code='LMGAO',
                 size=a.size,
                 minimum_area=a.minimum_area,
                 aug_mode='same',
                 normalize=True,
                 seed=a.seed,
            ) for t in ('train', 'test')
        ]

        out_dir = f'out/{a.experiment_name}/{config.code}/{config.model_name}_{a.source}'
        if a.suffix:
            out_dir += f'_{a.suffix}'

        trainer = Trainer(
            config=config,
            out_dir=out_dir,
            train_dataset=dss[0],
            val_dataset=dss[1],
            use_gpu=not a.cpu,
            overwrite=a.overwrite,
            experiment_name=a.experiment_name,
        )

        trainer.start(a.epoch)


if __name__ == '__main__':
    cli = CLI()
    cli.run()
