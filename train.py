import os
import re
from glob import glob
import base64
import hashlib

import numpy as np
import torch
from torch import nn
from torch import optim
# import torch_optimizer as optim2
from torchvision.utils import make_grid
from sklearn import metrics
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pydantic import Field
from timm.scheduler.cosine_lr import CosineLRScheduler

from endaaman.ml import BaseDLArgs, BaseMLCLI, BaseTrainer, BaseTrainerConfig
from endaaman.metrics import MultiAccuracy, AccuracyByChannel, BaseMetrics
from endaaman.functional import multi_accuracy

from models import TimmModel, AttentionModel, CrossEntropyLoss, NestedCrossEntropyLoss
from datasets import BrainTumorDataset, BatchedBrainTumorDataset, NUM_TO_DIAG



def label_to_text(result):
    ss = []
    for i, v in enumerate(result):
        label = NUM_TO_DIAG[i]
        ss.append(f'{label}:{v:.3f}')
    return '\n'.join(ss)


class LMGAccuracy(BaseMetrics):
    def __call__(self, preds, gts):
        summed = torch.sum(preds[:, [2, 3, 4]], dim=-1)
        preds[:, 2] = summed
        preds = preds[:, :3]
        gts = gts.clamp(max=2)
        return multi_accuracy(preds, gts, by_index=True)


class TrainerConfig(BaseTrainerConfig):
    code: str
    loss_weights: str
    use_mil: bool
    model_name:str
    crop_size: int
    input_size: int


class Trainer(BaseTrainer):
    def prepare(self):
        num_classes = len(set([*self.config.code]) - {'_'})
        if num_classes == 5:
            self.criterion = NestedCrossEntropyLoss(
                rules=[{
                    'weight': int(self.config.loss_weights[0]),
                    'index': [],
                }, {
                    'weight': int(self.config.loss_weights[1]),
                    'index': [[2, 3, 4]], # merge G,A,O to G
                }])
        else:
            self.criterion = CrossEntropyLoss(input_logits=True)
        if self.config.use_mil:
            return AttentionModel(name=self.config.model_name, num_classes=num_classes, params_count=10)
        return TimmModel(name=self.config.model_name, num_classes=num_classes)

    def eval(self, inputs, gts):
        if self.config.use_mil:
            inputs = inputs[0]
            gts = gts[0]
        preds = self.model(inputs.to(self.device), activate=False)
        loss = self.criterion(preds, gts.to(self.device))
        if self.config.use_mil:
            preds = preds[None, ...]
        return loss, preds.detach().cpu()



    def get_metrics(self):
        return {
            'acc': MultiAccuracy(),
            # 'acc3': LMGAccuracy(),
            # **{
            #     f'acc_{l}{NUM_TO_DIAG[l]}': AccuracyByChannel(target_channel=l) for l in range(self.num_classes)
            # },
        }

class CLI(BaseMLCLI):
    class CommonArgs(BaseDLArgs):
        lr: float = 0.0001
        batch_size: int = 8
        use_mil: bool = Field(False, cli=('--mil', ))
        num_workers: int = 4
        epoch: int = 50
        model_name: str = Field('tf_efficientnetv2_b0', cli=('--model', '-m'))
        suffix: str = ''
        crop_size: int = Field(512, cli=('--crop-size', '-c'))
        input_size: int = Field(512, cli=('--input-size', '-i'))
        size: int = Field(-1, cli=('--size', '-s'))
        code: str = 'LMGAO'
        loss_weights: str = '10'
        overwrite: bool = Field(False, cli=('--overwrite', '-o'))

    def run_start(self, a):
        config = TrainerConfig(
            batch_size=1 if a.use_mil else a.batch_size,
            num_workers=a.num_workers,
            lr=a.lr,
            code=a.code,
            use_mil=a.use_mil,
            model_name=a.model_name,
            loss_weights=a.loss_weights,
            crop_size=a.size if a.size > 0 else a.crop_size,
            input_size=a.size if a.size > 0 else a.input_size,
        )

        if a.use_mil:
            dss = [
                BatchedBrainTumorDataset(
                    target=t,
                    code=a.code,
                    aug_mode='same',
                    crop_size=config.crop_size,
                    input_size=config.input_size,
                    seed=a.seed,
                    batch_size=a.batch_size,
                ) for t in ('train', 'test')
            ]
        else:
            dss = [
                BrainTumorDataset(
                    target=t,
                    code=a.code,
                    aug_mode='same',
                    crop_size=config.crop_size,
                    input_size=config.input_size,
                    seed=a.seed,
                ) for t in ('train', 'test')
            ]

        subdir = f'{config.code}-MIL' if a.use_mil else config.code
        out_dir = f'out/models/{subdir}/{config.model_name}'
        if a.suffix:
            out_dir += f'_{a.suffix}'

        trainer = Trainer(
            config=config,
            out_dir=out_dir,
            train_dataset=dss[0],
            val_dataset=dss[1],
            use_gpu=not a.cpu,
            overwrite=a.overwrite,
        )

        trainer.start(a.epoch)



if __name__ == '__main__':
    cli = CLI()
    cli.run()
