import os
import re
from glob import glob

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

from endaaman.ml import pil_to_tensor, tensor_to_pil
from endaaman.ml2 import BaseDLArgs, BaseMLCLI
from endaaman.trainer2 import BaseTrainer, BaseTrainerConfig
from endaaman.metrics import MultiAccuracy, AccuracyByChannel, BaseMetrics
from endaaman.functional import multi_accuracy

from models import TimmModel, CrossEntropyLoss, NestedCrossEntropyLoss
from datasets import BrainTumorDataset, NUM_TO_DIAG



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
    model_name:str
    grid_size: int
    crop_size: int
    input_size: int


class Trainer(BaseTrainer):
    def prepare(self):
        num_classes = len(set([*self.config.code]) - {'_'})
        print(num_classes)
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
        return TimmModel(name=self.config.model_name, num_classes=num_classes)

    def eval(self, inputs, gts):
        preds = self.model(inputs.to(self.device), activate=False)
        loss = self.criterion(preds, gts.to(self.device))
        return loss, preds.detach()

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
        lr: float = 0.001
        model_name: str = Field('tf_efficientnetv2_b0', cli=('--model', '-m'))
        grid_size: int = Field(1024, cli=('--grid', '-g'))
        crop_size: int = Field(512, cli=('--crop', '-c'))
        input_size: int = Field(512, cli=('--input', '-i'))
        code: str = 'LMGAO'
        loss_weights: str = '10'

    def arg_start(self, parser):
        parser.add_argument('--model', '-m', default='tf_efficientnetv2_b0_5')

    def run_start(self, a):
        config = TrainerConfig(
            batch_size=a.batch_size,
            num_workers=a.num_workers,
            lr=a.lr,
            model_name=a.model_name,
            code=a.code,
            loss_weights=a.loss_weights,
            grid_size=a.grid_size,
            crop_size=a.crop_size,
            input_size=a.input_size,
        )

        dss = [
            BrainTumorDataset(
                target=t,
                code=a.code,
                aug_mode='same',
                grid_size=self.a.grid_size,
                crop_size=self.a.crop_size,
                input_size=self.a.input_size,
                seed=self.a.seed,
            ) for t in ('train', 'test')
        ]

        trainer = Trainer(
            config=config,
            out_dir=f'out/models/{config.code}/{config.model_name}',
            train_dataset=dss[0],
            val_dataset=dss[1],
            use_gpu=not a.cpu
        )

        trainer.start(a.epoch)



if __name__ == '__main__':
    cli = CLI()
    cli.run()
