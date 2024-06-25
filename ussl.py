import os
import re
import math

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from pydantic import Field
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import leidenalg as la
import igraph as ig

from endaaman.ml import BaseTrainer, BaseTrainerConfig, BaseMLCLI, Checkpoint
from endaaman.ml.utils import hover_images_on_scatters

from models import TimmModel, SimSiamModel, BarlowTwinsModel
from models.loss import SymmetricCosSimLoss, BarlowTwinsLoss
from datasets import MEAN, STD
from datasets.fold import PairedFoldDataset, FoldDataset


J = os.path.join


class TrainerConfig(BaseTrainerConfig):
    model_name:str = 'resnetrs50'
    source: str = 'enda4_512'
    code: str = 'LMGAOB'
    total_fold: int = 5
    fold: int = 0
    size: int = 256
    minimum_area: float = 0.6
    limit: int = 100
    noupsample: bool = False
    pretrained: bool = False
    scheduler: str = 'step_10'

    mean: float = MEAN
    std: float = STD

    arc: str = Field('simsiam', choices=['simsiam', 'barlow'])
    num_features: int = Field(2048, s='-f')
    no_stop_grads: bool = False

    def unique_code(self):
        return [c for c in dict.fromkeys(self.code) if c in 'LMGAOB']

    def num_classes(self):
        return len(self.unique_code())


class Trainer(BaseTrainer):
    def prepare(self):
        if self.config.arc == 'simsiam':
            model = SimSiamModel(
                name=self.config.model_name,
                num_features=self.config.num_features,
                pretrained=self.config.pretrained)
            self.criterion = SymmetricCosSimLoss(stop_grads=not self.config.no_stop_grads)
        elif self.config.arc == 'barlow':
            model = BarlowTwinsModel(
                name=self.config.model_name,
                num_features=self.config.num_features,
                pretrained=self.config.pretrained
            )
            self.criterion = BarlowTwinsLoss(lambd=5e-3)
        else:
            raise RuntimeError('Invalid arc', self.config.arc)
        return model

    def create_optimizer(self):
        return optim.RAdam(self.model.parameters())

    def create_scheduler(self):
        s = self.config.scheduler
        if not s:
            return None
        m = re.match(r'^step_(\d+)$', s)
        if m:
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=int(m[1]), gamma=0.1)
        raise RuntimeError('Invalid scheduler')

    def eval(self, inputs, __gts, i):
        inputs = inputs.to(self.device)
        if self.config.arc == 'simsiam':
            # inputs: [N, 2, C, H, W]
            z0, z1, p0, p1 = self.model(inputs)
            loss = self.criterion(z0, z1, p0, p1)
            return loss, p0

        if self.config.arc == 'barlow':
            # inputs: [N, 2, C, H, W]
            x0 = inputs[:, 0, ...]
            x1 = inputs[:, 1, ...]
            z0 = self.model(x0, normalize=True)
            z1 = self.model(x1, normalize=True)
            loss = self.criterion(z0, z1)
            return loss, z0

        raise RuntimeError('Invalid arc', self.config.arc)

    def metrics_std(self, preds, gts, batch):
        preds = F.normalize(preds, p=2, dim=1)
        std = torch.std(preds, dim=0).mean()
        return std.detach().cpu()


class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        pass

    class TrainArgs(CommonArgs, TrainerConfig):
        base_lr: float = 1e-5 # ViT for 1e-6
        lr: float = -1
        batch_size: int = Field(48, s='-B')
        num_workers: int = Field(4, s='-N')
        epoch: int = Field(30, s='-E')
        suffix: str = ''
        out: str = 'out/{arc}/fold{total_fold}_{fold}/{code}-{model_name}{suffix}'
        overwrite: bool = Field(False, s='-O')


    def run_train(self, a:TrainArgs):
        m = re.match(r'^.*_(\d+)$', a.source)
        assert m
        a.lr = a.lr if a.lr>0 else a.base_lr*a.batch_size

        config = TrainerConfig(**a.dict())

        dss = [PairedFoldDataset(
             source = a.source,
             total_fold = a.total_fold,
             fold = a.fold,
             target = t,
             code = a.code,
             size = a.size,
             minimum_area = a.minimum_area,
             limit = a.limit,
             upsample = not config.noupsample,
             augmentation = True,
             normalization = True,
        ) for t in ('train', 'test')]

        out_dir = a.out.format(**a.dict())

        trainer = Trainer(
            config=config,
            out_dir=out_dir,
            train_dataset=dss[0],
            val_dataset=dss[1],
            use_gpu=True,
            multi_gpu=True,
            overwrite=a.overwrite,
            experiment_name=a.source,
        )

        trainer.start(a.epoch)


    class ValidateArgs(CommonArgs):
        model_dir: str = Field(..., s='-d')
        target: str = Field('test', choices=['train', 'test', 'all'])
        batch_size: int = Field(128, s='-B')
        use: str = Field('last', choices=['best', 'last', 'none', 'pretrained'])

    def run_validate(self, a:ValidateArgs):
        config = TrainerConfig.from_file(J(a.model_dir, 'config.json'))
        if config.arc == 'simsiam':
            model = SimSiamModel(
                name=config.model_name,
                num_features=config.num_features,
                pretrained=a.use == 'pretrained'
            )
        elif config.arc == 'barlow':
            model = BarlowTwinsModel(
                name=config.model_name,
                num_features=config.num_features,
                pretrained=a.use == 'pretrained'
            )
        else:
            raise RuntimeError('Invalid arc:', config.arc)

        print('config:', config)
        if a.use == 'best':
            chp = 'checkpoint_best.pt'
        elif a.use == 'last':
            chp = 'checkpoint_last.pt'
        else:
            # none | pretrained
            chp = None

        if chp:
            checkpoint = Checkpoint.from_file(J(a.model_dir, chp))
            model.load_state_dict(checkpoint.model_state)
        model = model.cuda().eval()

        transform = transforms.Compose([
            transforms.CenterCrop(config.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

        ds = FoldDataset(
             total_fold=config.total_fold,
             fold=config.fold,
             source=config.source,
             target=a.target,
             code=config.code,
             size=config.size,
             minimum_area=-1,
             augmentation=False,
             normalization=True,
             limit=-1,
        )
        df = ds.df

        num_chunks = math.ceil(len(ds.df) / a.batch_size)
        tq = tqdm(range(num_chunks))
        featuress = []
        for chunk in tq:
            i0 = chunk*a.batch_size
            i1 = (chunk+1)*a.batch_size
            rows = df[i0:i1]
            tt = []
            for i, row in rows.iterrows():
                image = ds.load_from_row(row)
                tt.append(transform(image))
                image.close()

            tt = torch.stack(tt)
            with torch.set_grad_enabled(False):
                f = model.forward_features(tt.cuda(), use_mlp=True)
            features = f.detach().cpu()
            featuress.append(features)
            tq.set_description(f'{i0} - {i1}')
            tq.refresh()

        features = torch.cat(featuress)
        features = features.reshape(features.shape[0], features.shape[1])
        # torch.save(features, J(a.model_dir, f'features_{a.target}.pt'))

        data = [
            dict(zip(['name', 'filename', 'diag', 'diag_org', 'feature'], values))
            for values in zip(
                df['name'],
                df['filename'],
                df['diag'],
                df['diag_org'],
                features.numpy()
            )
        ]
        torch.save(data, J(a.model_dir, f'features_{a.use}_{a.target}.pt'))



if __name__ == '__main__':
    cli = CLI()
    cli.run()
