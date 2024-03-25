
import os
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
from pydantic import Field, Extra
# from timm.scheduler.cosine_lr import CosineLRScheduler
from vit_pytorch import ViT
# from vit_pytorch import Dino
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import pytorch_grad_cam as CAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget, ClassifierOutputTarget

from endaaman import with_wrote, load_images_from_dir_or_file, grid_split, with_mkdir
from endaaman.ml import BaseTrainerConfig, BaseTrainer, Checkpoint, pil_to_tensor
from endaaman.ml.metrics import MultiAccuracy
from endaaman.ml.cli import BaseMLCLI, BaseDLArgs, BaseTrainArgs

from models import TimmModel, CrossEntropyLoss
from datasets.fold import DinoFoldDataset, FoldDataset, MEAN, STD
from utils.dino import Dino


np.set_printoptions(suppress=True, floatmode='fixed')
J = os.path.join

class TrainerConfig(BaseTrainerConfig):
    source: str
    fold: int
    total_fold: int
    code: str
    num_classes: int
    size: int
    minimum_area: float
    limit: int = -1
    upsample: bool = False
    mean: float = MEAN
    std: float = STD

    def unique_code(self):
        return [c for c in dict.fromkeys(self.code) if c in 'LMGAOB']


class Trainer(BaseTrainer):
    def prepare(self):
        # self.criterion = CrossEntropyLoss(input_logits=True)
        # self.fig_col_count = 2
        model = ViT(
            image_size = 512,
            patch_size = 32,
            num_classes = self.config.num_classes,
            dim = 1024,
            depth = 6,
            heads = 8,
            mlp_dim = 2048
        )

        blur_limit = (3, 5)
        aug = A.Compose([
            A.RandomRotate90(p=1),
            A.Flip(p=0.5),

            # Blurs
            A.OneOf([
                A.MotionBlur(blur_limit=blur_limit),
                A.MedianBlur(blur_limit=blur_limit),
                A.GaussianBlur(blur_limit=blur_limit),
                A.Blur(blur_limit=blur_limit),
            ], p=1.0),
            # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=5, p=0.5),

            # Brightness
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Emboss(),
                # A.RandomBrightnessContrast(brightness_limit=0.1),
                A.RandomToneCurve(),
            ], p=1.0),

            # Color
            A.OneOf([
                A.RGBShift(),
                A.HueSaturationValue(sat_shift_limit=20),
            ], p=1.0),

            # Noise
            A.OneOf([
                A.ISONoise(),
                A.GaussNoise(),
                A.ImageCompression(quality_lower=50, quality_upper=100),
            ], p=1.0),

            # Transform
            A.OneOf([
                A.CoarseDropout(max_holes=16, min_holes=1,
                                max_height=32, max_width=32,
                                min_height=8, min_width=8, fill_value=0, p=1.0),
                A.RandomGridShuffle(grid=(2, 2)),
                A.RandomGridShuffle(grid=(3, 3)),
            ], p=1.0),
            A.Normalize(mean=self.config.mean, std=self.config.std),
            ToTensorV2()
        ])

        self.learner = Dino(
            model,
            image_size = 512,
            hidden_layer = 'to_latent',     # hidden layer name or index, from which to extract the embedding
            projection_hidden_size = 256,   # projector network hidden dimension
            projection_layers = 4,          # number of layers in projection network
            num_classes_K = self.config.num_classes,          # output logits dimensions (referenced as K in paper)
            student_temp = 0.9,             # student temperature
            teacher_temp = 0.04,            # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
            local_upper_crop_scale = 0.4,   # upper bound for local crop - 0.4 was recommended in the paper
            global_lower_crop_scale = 0.5,  # lower bound for global crop - 0.5 was recommended in the paper
            moving_average_decay = 0.9,     # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
            center_moving_average_decay = 0.9, # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok

            augment_fn=lambda xx: torch.stack([aug(image=x.numpy())['image'] for x in xx]),
            augment_fn2=lambda xx: torch.stack([aug(image=x.numpy())['image'] for x in xx]),
        ).to(self.device)
        return model


    def select_inputs_and_gts(self, params):
        return params[0], None

    def eval(self, *params):
        pass

    def _eval(self, params, is_training):
        self.model.to(self.device)
        with torch.set_grad_enabled(is_training):
            # loss, preds = self.eval(*params)
            inputs, gts = params
            loss = self.learner(inputs.to(self.device))
        if is_training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.learner.update_moving_average()
        return loss, None

    # def _visualize_confusion(self, ax, label, preds, gts):
    #     preds = torch.argmax(preds, dim=-1)
    #     gts = gts.flatten()
    #     cm = skmetrics.confusion_matrix(gts.numpy(), preds.numpy())
    #     ticks = [*self.train_dataset.unique_code]
    #     sns.heatmap(cm, annot=True, ax=ax, fmt='g', xticklabels=ticks, yticklabels=ticks)
    #     ax.set_title(label)
    #     ax.set_xlabel('Predict', fontsize=13)
    #     ax.set_ylabel('GT', fontsize=13)

    # def visualize_train_confusion(self, ax, train_preds, train_gts, val_preds, val_gts):
    #     self._visualize_confusion(ax, 'train', train_preds, train_gts)

    # def visualize_val_confusion(self, ax, train_preds, train_gts, val_preds, val_gts):
    #     if val_preds is None or val_gts is None:
    #         return
    #     self._visualize_confusion(ax, 'val', val_preds, val_gts)

    # def create_scheduler(self):
    #     return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10)
    #     # return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.1)

    def continues(self):
        lr = self.get_current_lr()
        return lr > 1e-7

class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        pass

    class TrainArgs(BaseTrainArgs):
        source: str = 'enda4_512'
        batch_size: int = Field(16, s='-B')
        lr: float = 0.0001
        code: str = 'LMGGG_'
        total_fold: int = 5
        fold: int = 0
        minimum_area: float = 0.6
        limit: int = 500
        noupsample: bool = False

        num_workers: int = Field(4, s='-N')
        epoch: int = Field(100, s='-E')
        suffix: str = Field('', s='-S')
        prefix: str = ''
        overwrite: bool = Field(False, s='-O')

    def run_train(self, a:TrainArgs):
        num_classes = len(set([*a.code]) - {'_'})
        m = re.match(r'^.*_(\d+)$', a.source)
        assert m
        size = int(m[1])

        config = TrainerConfig(
            source = a.source,
            batch_size = a.batch_size,
            size = size,
            lr = a.lr,
            code = a.code,
            num_classes = num_classes,
            total_fold = a.total_fold,
            fold = a.fold,
            minimum_area = a.minimum_area,
            limit = a.limit,
            upsample = not a.noupsample,
        )

        if a.fold < 0:
            dss = [DinoFoldDataset(
                 source_dir = J('data/tiles', a.source),
                 total_fold = a.total_fold,
                 fold = -1,
                 target = 'all',
                 code = a.code,
                 size = size,
                 minimum_area = a.minimum_area,
                 limit = a.limit,
                 upsample = config.upsample,
                 augmentation = True,
                 normalization = True,
            ), None]
        else:
            dss = [
                DinoFoldDataset(
                    source_dir = J('data/tiles', a.source),
                    total_fold = a.total_fold,
                    fold = a.fold,
                    target = t,
                    code = a.code,
                    size = size,
                    minimum_area = a.minimum_area,
                    limit = a.limit,
                    upsample = config.upsample and t=='train',
                    augmentation= t=='train',
                    normalization = True,
                ) for t in ('train', 'test')
            ]

        out_dir = J(
            'out', f'{a.source}_dino', config.code,
            f'fold{a.total_fold}_{a.fold}', a.prefix, 'vit'
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
