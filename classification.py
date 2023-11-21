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

import pytorch_grad_cam as CAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget, ClassifierOutputTarget

from endaaman import with_wrote, load_images_from_dir_or_file
from endaaman.ml import BaseTrainerConfig, BaseTrainer, Checkpoint, pil_to_tensor
from endaaman.ml.metrics import MultiAccuracy
from endaaman.ml.cli import BaseMLCLI, BaseDLArgs, BaseTrainArgs

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

    def unique_code(self):
        return [c for c in dict.fromkeys(self.code) if c in 'LMGAOB']


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
        return lr > 1e-6

    def get_metrics(self):
        return {
            'acc': MultiAccuracy(),
        }

class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        pass

    class TrainArgs(BaseTrainArgs):
        lr: float = 0.01
        batch_size: int = Field(16, cli=('--batch-size', '-B', ))
        num_workers: int = 4
        minimum_area: float = 0.7
        limit: int = -1
        upsample: bool = Field(False, cli=('--upsample', ))
        noaug: bool = Field(False, cli=('--noaug', ))
        epoch: int = Field(100, cli=('--epoch', '-E'))
        total_fold: int = Field(..., cli=('--total-fold', ))
        fold: int
        model_name: str = Field('tf_efficientnet_b0', cli=('--model', '-m'))
        source: str = Field('enda3_512', cli=('--source', ))
        suffix: str = ''
        prefix: str = ''
        size: int = Field(512, cli=('--size', '-s'))
        code: str = 'LMGGGB'
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
            image_aug = not a.noaug,
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
                 image_aug=not a.noaug,
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

    class ValidateArgs(BaseDLArgs):
        model_dir: str = Field(..., cli=('--model-dir', '-d'))
        target: str = 'test'
        batch_size: int = Field(16, cli=('--batch-size', '-B', ))

    def run_validate(self, a:ValidateArgs):
        checkpoint:Checkpoint = torch.load(J(a.model_dir, 'checkpoint_last.pt'), map_location='cpu')
        config = TrainerConfig.from_file(J(a.model_dir, 'config.json'))
        model = TimmModel(name=config.model_name, num_classes=config.num_classes)
        model.load_state_dict(checkpoint.model_state)
        model.to(a.device())
        model = model.eval()

        transform = transforms.Compose([
            transforms.CenterCrop(config.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.7, std=0.1),
        ])

        ds = FoldDataset(
             total_fold=config.total_fold,
             fold=config.fold,
             source_dir=J('cache', config.source),
             target=a.target,
             code=config.code,
             size=config.size,
             minimum_area=-1,
             aug_mode='same',
             normalize=True,
        )

        df = ds.df.copy()
        df[ds.unique_code] = -1.0
        df['pred'] = '_'

        num_chunks = math.ceil(len(df) / a.batch_size)
        t = tqdm(range(num_chunks))

        ff = []
        for chunk in t:
            i0 = chunk*a.batch_size
            i1 = (chunk+1)*a.batch_size
            rows = df[i0:i1]
            tt = []
            for i, row in rows.iterrows():
                fp = J(f'cache', config.source,  row['diag_org'], row['name'], row['filename'])
                tt.append(transform(Image.open(fp)))

            tt = torch.stack(tt)
            with torch.set_grad_enabled(False):
                o, f = model(tt.to(a.device()), activate=True, with_feautres=True)
                o = o.detach().cpu().numpy()
                f = f.detach().cpu().numpy()
            df.loc[df.index[i0:i1], ds.unique_code] = o
            df.loc[df.index[i0:i1], 'pred'] = [ds.unique_code[i] for i in np.argmax(o, axis=1)]
            ff.append(f)
            t.set_description(f'{i0} - {i1}')
            t.refresh()

        df.to_excel(with_wrote(J(a.model_dir, f'{a.target}.xlsx')))

        features = np.concatenate(ff)
        names = list(df['name'] + '_' + df['order'].astype(str) + '_' + df['x'].astype(str) + '_' + df['y'].astype(str))
        feature_list = list(zip(names, features))
        torch.save(feature_list, J(a.model_dir, f'{a.target}_features.pt'))
        # for name, feature in zip(names, features):
        #     np.save(J(a.model_dir, 'features', name), feature)


    class PredictArgs(BaseDLArgs):
        model_dir: str = Field(..., cli=('--model-dir', '-d'))
        src: str
        cam: bool = Field(False, cli=('--cam', '-c'))
        cam_label: str = Field('', cli=('--cam-label', ))
        show: bool = Field(False, cli=('--show', ))
        crop: int = -1
        cols: int = 3

    def run_predict(self, a:ValidateArgs):
        checkpoint:Checkpoint = torch.load(J(a.model_dir, 'checkpoint_last.pt'), map_location='cpu')
        config = TrainerConfig.from_file(J(a.model_dir, 'config.json'))
        model = TimmModel(name=config.model_name, num_classes=config.num_classes)
        model.load_state_dict(checkpoint.model_state)
        model.to(a.device())
        model = model.eval()

        gradcam = CAM.GradCAM(
            model=model,
            target_layers=model.get_cam_layers(),
            use_cuda=not a.cpu) if a.cam else None

        image_transform = transforms.CenterCrop(a.crop) if a.crop > 0 else lambda x:x

        transform = transforms.Compose([v for v in [
            transforms.ToTensor(),
            transforms.Normalize(mean=0.7, std=0.1),
        ] if v])

        images, paths  = load_images_from_dir_or_file(a.src, with_path=True)
        images = [image_transform(i) for i in images]

        rows = math.ceil(len(images) / a.cols)
        fig = plt.figure(figsize=(a.cols*4, rows*3))

        for i, (path, image) in enumerate(zip(paths, images)):
            t = transform(image)[None, ...].to(a.device())
            with torch.set_grad_enabled(False):
                o = model(t, activate=True)
                o = o.detach().cpu().numpy()[0]

            unique_code = config.unique_code()
            pred_id = np.argmax(o)
            pred = unique_code[pred_id]
            label = ' '.join([f'{c}:{int(v*100):3d}' for c, v in zip(unique_code, o)])
            h, w = t.shape[2], t.shape[3]
            print(f'{path}: ({w}x{h}) {pred} ({label})')

            if not a.cam:
                continue

            cam_class_id = unique_code.index(a.cam_label) if a.cam_label  else pred_id
            cam_class = unique_code[cam_class_id]
            targets = [ClassifierOutputTarget(cam_class_id)]

            mask = gradcam(input_tensor=t, targets=targets)[0]
            vis = show_cam_on_image(np.array(image)/255, mask, use_rgb=True)
            vis = Image.fromarray(vis)
            d = J(a.model_dir, 'cam')
            os.makedirs(d, exist_ok=True)
            name = os.path.splitext(os.path.basename(path))[0]
            # vis.save(with_wrote(J(d, f'{name}_{cam_class}.jpg')))
            vis.save(J(d, f'{name}_{cam_class}.jpg'))

            if not a.show:
                continue
            ax = fig.add_subplot(rows, a.cols, i+1)
            ax.imshow(vis)
            ax.set_title(f'{name} {pred} (CAM: {cam_class})')
            ax.set(xlabel=None, ylabel=None)

        plt.show()



if __name__ == '__main__':
    cli = CLI()
    cli.run()
