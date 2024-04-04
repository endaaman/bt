
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

from models import TimmModel, CrossEntropyLoss, get_pool
from datasets.fold import DinoFoldDataset, FoldDataset, MEAN, STD
from utils.dino import Dino


def unique_code(code):
    return [c for c in dict.fromkeys(code) if c in 'LMGAOB']

np.set_printoptions(suppress=True, floatmode='fixed')
J = os.path.join

class TrainerConfig(BaseTrainerConfig):
    source: str
    model_name: str
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
        return unique_code(self.code)


class Trainer(BaseTrainer):
    def prepare(self):
        # self.criterion = CrossEntropyLoss(input_logits=True)
        # self.fig_col_count = 2

        if self.config.model_name == 'vit':
            model = ViT(
                image_size = self.config.size,
                patch_size = 32,
                num_classes = self.config.num_classes,
                dim = 1024,
                depth = 6,
                heads = 8,
                mlp_dim = 2048
            )
            hidden_layer = 'to_latent'
        else:
            model = TimmModel(name=self.config.model_name, num_classes=self.config.num_classes)
            hidden_layer = get_pool(model.base)

        self.learner = Dino(
            model,
            local_image_size = self.config.size//2,
            global_image_size = self.config.size,
            hidden_layer = hidden_layer,     # hidden layer name or index, from which to extract the embedding
            projection_hidden_size = 256,   # projector network hidden dimension
            projection_layers = 4,          # number of layers in projection network
            num_classes_K = 65336,          # output logits dimensions (referenced as K in paper)
            student_temp = 0.9,             # student temperature
            teacher_temp = 0.04,            # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
            local_upper_crop_scale = 0.4,   # upper bound for local crop - 0.4 was recommended in the paper
            global_lower_crop_scale = 0.5,  # lower bound for global crop - 0.5 was recommended in the paper
            moving_average_decay = 0.9,     # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
            center_moving_average_decay = 0.9, # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
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
            x1, x2, gts = params
            loss = self.learner(x1.to(self.device), x2.to(self.device))
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
        model_name: str = Field('vit', l='--model', s='-m')
        batch_size: int = Field(16, s='-B')
        code: str = 'LMGGG_'
        base_lr: float = 1e-5 # ViT for 1e-6
        lr: float = -1
        total_fold: int = 5
        fold: int = 0
        minimum_area: float = 0.6
        limit: int = 500
        noupsample: bool = False
        num_classes: int = Field(3, s='-C')

        num_workers: int = Field(4, s='-N')
        epoch: int = Field(100, s='-E')
        suffix: str = Field('', s='-S')
        prefix: str = ''
        overwrite: bool = Field(False, s='-O')

    def run_train(self, a:TrainArgs):
        m = re.match(r'^.*_(\d+)$', a.source)
        assert m
        size = int(m[1])
        lr = a.lr if a.lr>0 else a.base_lr*a.batch_size

        config = TrainerConfig(
            source = a.source,
            model_name = a.model_name,
            batch_size = a.batch_size,
            size = size,
            lr = lr,
            code = a.code,
            num_classes = a.num_classes,
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
            f'fold{a.total_fold}_{a.fold}', a.prefix, config.model_name,
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


    class ValidateArgs(BaseDLArgs):
        model_dir: str = Field(..., s='-d')
        target: str = Field('test', choices=['train', 'test', 'all'])
        batch_size: int = Field(16, s='-B')
        no_features: bool = False
        use_last: bool = False

    def run_validate(self, a:ValidateArgs):
        chp = 'checkpoint_last.pt' if a.use_last else 'checkpoint_best.pt'
        checkpoint = Checkpoint.from_file(J(a.model_dir, chp))
        config = TrainerConfig.from_file(J(a.model_dir, 'config.json'))
        model = TimmModel(name=config.model_name, num_classes=config.num_classes)
        model.to(a.device())
        model = model.eval()

        transform = transforms.Compose([
            transforms.CenterCrop(config.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.mean, std=config.std),
        ])

        ds = FoldDataset(
             total_fold=config.total_fold,
             fold=config.fold,
             source_dir=J('cache', config.source),
             target=a.target,
             code=config.code,
             size=config.size,
             minimum_area=-1,
             augmentation=False,
             normalization=True,
             limit=config.limit,
        )

        df = ds.df.copy()
        df[ds.unique_code] = -1.0
        df['pred'] = '_'

        batch_count = math.ceil(len(df) / a.batch_size)
        tq = tqdm(range(batch_count))

        embss = []
        featuress = []
        for chunk in tq:
            i0 = chunk*a.batch_size
            i1 = (chunk+1)*a.batch_size
            rows = df[i0:i1]
            tt = []
            for i, row in rows.iterrows():
                fp = J(f'cache', config.source,  row['diag_org'], row['name'], row['filename'])
                image = Image.open(fp)
                tt.append(transform(image))
                image.close()

            tt = torch.stack(tt)
            with torch.set_grad_enabled(False):
                i = tt.to(a.device())
                if a.no_features:
                    embs = model(i, activate=False, with_feautres=False)
                else:
                    embs, f = model(i, activate=False, with_feautres=True)
                    features = f.detach().cpu().numpy()
                    featuress.append(features)
                embs = embs.detach().cpu().numpy()

            embss.append(embs)

            tq.set_description(f'{i0} - {i1}')
            tq.refresh()
        # df.to_excel(with_wrote(J(a.model_dir, f'validate_{target}.xlsx')))

        embs = np.concatenate(embss)
        data = [
            dict(zip(['name', 'filename', 'diag', 'diag_org', 'pred', 'feature'], values))
            for values in zip(
                df['name'],
                df['filename'],
                df['diag'],
                df['diag_org'],
                df['pred'],
                embs
            )
        ]
        torch.save(data, J(a.model_dir, f'embs_{a.target}.pt'))

        if not a.no_features:
            features = np.concatenate(featuress)
            features = features.reshape(features.shape[0], features.shape[1])
            data = [
                dict(zip(['name', 'filename', 'diag', 'diag_org', 'pred', 'feature'], values))
                for values in zip(
                    df['name'],
                    df['filename'],
                    df['diag'],
                    df['diag_org'],
                    df['pred'],
                    features
                )
            ]
            torch.save(data, J(a.model_dir, f'features_{a.target}.pt'))




if __name__ == '__main__':
    cli = CLI()
    cli.run()
