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
from pydantic import Field
# from timm.scheduler.cosine_lr import CosineLRScheduler

import pytorch_grad_cam as CAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget, ClassifierOutputTarget

from endaaman import with_wrote, load_images_from_dir_or_file, grid_split
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
        default: bool = Field(False, cli=('--default', ))
        features: bool = Field(False, cli=('--features', '-F'))

    def run_validate(self, a:ValidateArgs):
        checkpoint:Checkpoint = torch.load(J(a.model_dir, 'checkpoint_last.pt'), map_location='cpu')
        config = TrainerConfig.from_file(J(a.model_dir, 'config.json'))
        model = TimmModel(name=config.model_name, num_classes=config.num_classes)
        if not a.default:
            model.load_state_dict(checkpoint.model_state)
        model.to(a.device())
        model = model.eval()

        transform = transforms.Compose([
            transforms.CenterCrop(config.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.7, std=0.2),
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

        target = 'un' + a.target if a.default else a.target

        df = ds.df.copy()
        df[ds.unique_code] = -1.0
        df['pred'] = '_'

        num_chunks = math.ceil(len(df) / a.batch_size)
        tq = tqdm(range(num_chunks))

        ff = []
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
                if a.features:
                    o, f = model(i, activate=True, with_feautres=True)
                    f = f.detach().cpu().numpy()
                    ff.append(f)
                else:
                    o = model(i, activate=True, with_feautres=False)
                o = o.detach().cpu().numpy()
            df.loc[df.index[i0:i1], ds.unique_code] = o
            preds = [ds.unique_code[i] for i in np.argmax(o, axis=1)]
            df.loc[df.index[i0:i1], 'pred'] = preds

            diags = df.loc[df.index[i0:i1], 'diag']
            acc = np.mean(diags == preds)
            tq.set_description(f'{i0} - {i1}: Acc: {acc:.3f}')
            tq.refresh()

        df.to_excel(with_wrote(J(a.model_dir, f'{target}.xlsx')))

        if a.features:
            features = np.concatenate(ff)
            names = list(df['name'] + '_' + df['order'].astype(str) + '_' + df['x'].astype(str) + '_' + df['y'].astype(str))
            feature_list = list(zip(names, features))
            torch.save(feature_list, J(a.model_dir, f'{target}_features.pt'))


    class PredictArgs(BaseDLArgs):
        model_dir: str = Field(..., cli=('--model-dir', '-d'))
        src: str
        cam: bool = Field(False, cli=('--cam', '-c'))
        cam_label: str = Field('', cli=('--cam-label', ))
        show: bool = Field(False, cli=('--show', ))
        size: int = -1
        grid: int = -1
        cols: int = 3

    def run_predict(self, a:PredictArgs):
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


        transform = transforms.Compose([v for v in [
            transforms.ToTensor(),
            transforms.Normalize(mean=0.7, std=0.2),
        ] if v])

        images, paths  = load_images_from_dir_or_file(a.src, with_path=True)
        imagess = []
        col_counts = []
        row_counts = []
        if a.grid > 0:
            for image in images:
                ggg = grid_split(image, size=a.grid,  overwrap=False, flattern=False)
                ii = []
                for gg in ggg:
                    for g in gg:
                        ii.append(g)
                col_counts.append(len(ggg[0]))
                row_counts.append(len(ggg))
                imagess.append(ii)
        else:
            if a.size > 0:
                crop = transforms.CenterCrop(a.size)
                for image in images:
                    imagess.append([crop(image)])
            else:
                for image in images:
                    imagess.append([image])

        rows = math.ceil(len(images) / a.cols)
        cols = min(a.cols, len(images))

        fig = plt.figure(figsize=(cols*4, rows*3))

        unique_code = config.unique_code()

        colors = {
            'L': 'green',
            'M': 'blue',
            'G': 'red',
            'A': 'yellow',
            'O': 'purple',
            'B': 'black',
        }

        font = ImageFont.truetype('/usr/share/fonts/ubuntu/Ubuntu-R.ttf', size=16)

        print(imagess)

        for idx, (path, images, col_count, row_count) in enumerate(zip(paths, imagess, col_counts, row_counts)):
            oo = []
            drews = [[None]*col_count]*row_count

            for i, image in enumerate(images):
                t = transform(image)[None, ...].to(a.device())
                with torch.set_grad_enabled(False):
                    o = model(t, activate=True)
                    o = o.detach().cpu().numpy()[0]
                oo.append(o)

                pred = unique_code[np.argmax(o)]

                # image = image.copy()
                if a.cam:
                    cam_class_id = unique_code.index(a.cam_label) if a.cam_label  else pred_id
                    cam_class = unique_code[cam_class_id]
                    targets = [ClassifierOutputTarget(cam_class_id)]

                    mask = gradcam(input_tensor=t, targets=targets)[0]
                    vis = show_cam_on_image(np.array(image)/255, mask, use_rgb=True)
                    # overwrite variable
                    image = Image.fromarray(vis)

                draw = ImageDraw.Draw(image)
                draw.rectangle(
                    xy=((0, 0), (image.width, image.height)),
                    outline=colors[pred],
                )
                text = ' '.join([f'{k}:{v:.2f}' for v, k in zip(o, unique_code)])
                bb = draw.textbbox(xy=(0, 0), text=text, font=font, spacing=8)
                draw.rectangle(
                    xy=bb,
                    fill=colors[pred],
                )
                draw.text(
                    xy=(0, 0),
                    text=text,
                    font=font,
                    fill='white'
                )
                print(i, i // col_count, i % col_count, image)
                drews[i // col_count][i % col_count] = image

            print(drews[0])
            print(drews[1])
            # merged_image =Image.fromarray(cv2.vconcat(cv2.hconcat(drews[0]), cv2.hconcat(drews[1])))

            merged_image = image

            # row_images = []
            # for y, row in enumerate(drews):
            #     print(y, row)
            #     row_image_list = []
            #     for x, d in enumerate(row):
            #         print(d)
            #         row_image_list.append(np.array(d))
            #     h = cv2.hconcat(row_image_list)
            #     row_images.append(h)
            # merged_image = Image.fromarray(cv2.vconcat(row_images))

            o = np.stack(oo).mean(axis=0)

            pred_id = np.argmax(o)
            pred = unique_code[pred_id]
            label = ' '.join([f'{c}:{int(v*100):3d}' for c, v in zip(unique_code, o)])
            h, w = t.shape[2], t.shape[3]
            print(f'{path}: ({w}x{h}) {pred} ({label})')

            if not a.show:
                continue

            name = os.path.splitext(os.path.basename(path))[0]
            ax = fig.add_subplot(rows, cols, idx+1)
            ax.imshow(merged_image)
            title = f'{name} {pred}'
            if a.cam:
                title += f'(CAM: {cam_class})'
            ax.set_title(title)
            ax.set(xlabel=None, ylabel=None)

        if a.show:
            plt.show()



if __name__ == '__main__':
    cli = CLI()
    cli.run()
