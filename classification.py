import os
import math
import json
import re
from glob import glob

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
# import pytorch_grad_cam as CAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from endaaman import with_wrote
from endaaman.ml.metrics import MultiAccuracy
from endaaman.ml import BaseMLCLI, BaseDLArgs, BaseTrainerConfig, BaseTrainer, Checkpoint, pil_to_tensor

from models import TimmModel, CrossEntropyLoss
from datasets.fold import FoldDataset


np.set_printoptions(suppress=True, floatmode='fixed')
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
    limit: int = -1
    upsample: bool = False
    image_aug: bool = False


class Trainer(BaseTrainer):
    def prepare(self):
        self.criterion = CrossEntropyLoss(input_logits=True)
        self.fig_col_count = 2
        return TimmModel(name=self.config.model_name, num_classes=self.config.num_classes)

    def eval(self, inputs, gts):
        preds = self.model(inputs.to(self.device), activate=False)
        loss = self.criterion(preds, gts.to(self.device))
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

    def create_scheduler(self):
        return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)

    def continues(self):
        lr = self.get_current_lr()
        return lr > 1e-5

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
        limit: int = -1
        upsample: bool = Field(False, cli=('--upsample', ))
        aug: bool = Field(False, cli=('--aug', ))
        epoch: int = Field(50, cli=('--epoch', '-E'))
        total_fold: int = Field(6, cli=('--total-fold', ))
        fold: int = 0
        model_name: str = Field('tf_efficientnet_b0', cli=('--model', '-m'))
        source: str = Field('enda2_512', cli=('--source', ))
        suffix: str = ''
        prefix: str = ''
        size: int = Field(512, cli=('--size', '-s'))
        code: str = 'LMGAO'
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
            limit = a.limit,
            upsample = a.upsample,
            image_aug = a.aug,
        )

        dss = [
            FoldDataset(
                 total_fold=a.total_fold,
                 fold=a.fold,
                 source_dir=J('cache', a.source),
                 target=t,
                 code=a.code,
                 size=a.size,
                 minimum_area=a.minimum_area,
                 limit=a.limit,
                 upsample = a.upsample,
                 image_aug=a.aug,
                 aug_mode='same',
                 normalize=True,
            ) for t in ('train', 'test')
        ]

        out_dir = J(
            'out', a.source, config.code,
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

    class ValidateArgs(CommonArgs):
        model_dir: str = Field(..., cli=('--model-dir', '-d'))

        target: str = 'test'
        batch_size: int = Field(16, cli=('--batch-size', '-B', ))
        total_fold: int = Field(6, cli=('--total-fold', ))
        fold: int = 0
        source: str = Field('enda2_512', cli=('--source', ))
        size: int = Field(512, cli=('--size', '-s'))

    def run_validate(self, a:ValidateArgs):
        checkpoint:Checkpoint = torch.load(J(a.model_dir, 'checkpoint_best.pt'))
        config = TrainerConfig.from_file(J(a.model_dir, 'config.json'))
        model = TimmModel(name=config.model_name, num_classes=config.num_classes)
        model.load_state_dict(checkpoint.model_state)
        model.to(a.device())
        model = model.eval()

        transform = transforms.Compose([
            transforms.CenterCrop(a.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.7, std=0.1),
        ])

        ds = FoldDataset(
             total_fold=a.total_fold,
             fold=a.fold,
             source_dir=J('cache', a.source),
             target=a.target,
             code=config.code,
             size=a.size,
             minimum_area=-1,
             aug_mode='same',
             normalize=True,
        )

        df = ds.df.copy()
        if a.target == 'train':
            df = df.iloc[:1000]
        df[ds.unique_code] = -1.0
        df['pred'] = ''

        num_chunks = math.ceil(len(df) / a.batch_size)
        t = tqdm(range(num_chunks))

        for chunk in t:
            i0 = chunk*a.batch_size
            i1 = (chunk+1)*a.batch_size
            rows = df[i0:i1]
            tt = []
            for i, row in rows.iterrows():
                fp = J(f'cache/{a.source}', row['diag_org'], row['name'], row['filename'])
                tt.append(transform(Image.open(fp)))

            tt = torch.stack(tt)
            with torch.set_grad_enabled(False):
                o = model(tt.to(a.device()), activate=True).detach().cpu().numpy()
            df.loc[df.index[i0:i1], ds.unique_code] = o
            df.loc[df.index[i0:i1], 'pred'] = [ds.unique_code[i] for i in np.argmax(o, axis=1)]
            t.set_description(f'{i0} - {i1}')
            t.refresh()

        df.to_excel(with_wrote(J(a.model_dir, f'{a.target}.xlsx')))


if __name__ == '__main__':
    cli = CLI()
    cli.run()
