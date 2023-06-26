import os
import json
import re
from glob import glob
import base64
import hashlib

import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import transforms
import seaborn as sns
# import torch_optimizer as optim2
from torchvision.utils import make_grid
from sklearn import metrics as skmetrics
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pydantic import Field
from timm.scheduler.cosine_lr import CosineLRScheduler

from endaaman import load_images_from_dir_or_file
from endaaman.ml import BaseDLArgs, BaseMLCLI, BaseTrainer, BaseTrainerConfig, pil_to_tensor
from endaaman.ml.metrics import MultiAccuracy, AccuracyByChannel, BaseMetrics
from endaaman.ml.functional import multi_accuracy

from models import TimmModel, AttentionModel, CrossEntropyLoss, NestedCrossEntropyLoss
from datasets import BrainTumorDataset, BatchedBrainTumorDataset, NUM_TO_DIAG, MEAN, STD

J = os.path.join


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
    source: str


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

    def _visualize_confusion(self, ax, label, preds, gts):
        preds = torch.argmax(preds, dim=-1)
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
            # 'acc3': LMGAccuracy(),
            # **{
            #     f'acc_{l}{NUM_TO_DIAG[l]}': AccuracyByChannel(target_channel=l) for l in range(self.num_classes)
            # },
        }

class CLI(BaseMLCLI):
    class CommonArgs(BaseDLArgs):
        pass

    class TrainArgs(CommonArgs):
        lr: float = 0.0002
        batch_size: int = Field(16, cli=('--batch-size', '-B', ))
        use_mil: bool = Field(False, cli=('--mil', ))
        num_workers: int = 4
        epoch: int = 10
        model_name: str = Field('tf_efficientnetv2_b0', cli=('--model', '-m'))
        source: str = Field('images', cli=('--source', ))
        suffix: str = ''
        crop_size: int = Field(512, cli=('--crop-size', '-c'))
        input_size: int = Field(512, cli=('--input-size', '-i'))
        size: int = Field(-1, cli=('--size', '-s'))
        code: str = 'LMGAO'
        loss_weights: str = '10'
        experiment_name:str = Field('Default', cli=('--exp', ))
        overwrite: bool = Field(False, cli=('--overwrite', '-o'))

    def run_train(self, a:TrainArgs):
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
            source=a.source,
        )

        source_dir = J('datasets/LMGAO', a.source)

        if a.use_mil:
            dss = [
                BatchedBrainTumorDataset(
                    target=t,
                    source_dir=source_dir,
                    code=a.code,
                    aug_mode='same',
                    crop_size=config.crop_size,
                    input_size=config.input_size,
                    batch_size=a.batch_size,
                    seed=a.seed,
                ) for t in ('train', 'test')
            ]
        else:
            dss = [
                BrainTumorDataset(
                    target=t,
                    source_dir=source_dir,
                    code=a.code,
                    aug_mode='same',
                    crop_size=config.crop_size,
                    input_size=config.input_size,
                    seed=a.seed,
                ) for t in ('train', 'test')
            ]

        subdir = f'{config.code}-MIL' if a.use_mil else config.code
        out_dir = f'out/{a.experiment_name}/{subdir}/{config.model_name}_{a.source}'
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

    class PredArgs(CommonArgs):
        src: str
        model_dir: str = Field(..., cli=('--model-dir', '-M'))

    def run_pred(selfl, a):
        with open(J(a.model_dir, 'config.json')) as f:
            config = TrainerConfig(**json.load(f))
        model = TimmModel(name=config.model_name, num_classes=5)
        checkpoint = torch.load(J(a.model_dir, 'checkpoint_best.pt'))
        model.load_state_dict(checkpoint.model_state)

        ii, pp = load_images_from_dir_or_file(a.src, with_path=True)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

        for i, p in zip(ii, pp):
            # t = pil_to_tensor(i)[None, ...]
            t = transform(i)[None, ...]
            pred = model(t, activate=True)
            print(p, NUM_TO_DIAG[torch.argmax(pred.flatten())])



if __name__ == '__main__':
    cli = CLI()
    cli.run()
