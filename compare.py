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

import pytorch_grad_cam as CAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget, ClassifierOutputTarget

from endaaman import with_wrote, load_images_from_dir_or_file, grid_split, with_mkdir
from endaaman.ml import BaseTrainerConfig, BaseTrainer, Checkpoint, pil_to_tensor
from endaaman.ml.metrics import MultiAccuracy, BaseMetrics
from endaaman.ml.functional import multi_accuracy
from endaaman.ml.cli import BaseMLCLI, BaseDLArgs

from models import CompareModel
from datasets import FoldDataset, MEAN, STD
from utils import draw_frame


torch.multiprocessing.set_sharing_strategy('file_system')
np.set_printoptions(suppress=True, floatmode='fixed')
J = os.path.join


class TrainerConfig(BaseTrainerConfig):
    # model
    base: str = Field('uni', choices=[
        'baseline-vit'
        'baseline-cnn'
        'gigapath',
        'uni',
        'ctranspath',
    ],)
    source: str = 'enda4_512'

    # dataset
    total_fold: int = 5
    fold: int = 0
    code: str = 'LMGAOB'
    crop_size: int = 512
    size: int = 224
    minimum_area: float = 0.6
    limit: int = 100
    noupsample: bool = False

    # training
    lr: float = 0.0001
    scheduler: str = 'static'
    encoder = Field('frozen', choices=['frozen', 'unfrozen'])

    def unique_code(self):
        return [c for c in dict.fromkeys(self.code) if c in 'LMGAOB']

    def num_classes(self):
        return len(self.unique_code())

class Acc3(BaseMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mat = torch.tensor([
            # L, M, G, B
            [ 1, 0, 0, 0, ], # L
            [ 0, 1, 0, 0, ], # M
            [ 0, 0, 1, 0, ], # G
            [ 0, 0, 1, 0, ], # A
            [ 0, 0, 1, 0, ], # O
            [ 0, 0, 0, 1, ], # B
        ]).float()
        self.I = torch.tensor([0, 1, 2, 2, 2, 3])

    def calc(self, preds, gts, batch):
        num_classes = preds.shape[-1]
        new_preds = torch.matmul(preds, self.mat[:num_classes, :].to(preds.device))
        new_gts = self.I.to(gts.device)[gts]
        return multi_accuracy(new_preds, new_gts)


class Acc4(BaseMetrics):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mat = torch.tensor([
            # L, M, G,AO, B
            [ 1, 0, 0, 0, 0,], # L
            [ 0, 1, 0, 0, 0,], # M
            [ 0, 0, 1, 0, 0,], # G
            [ 0, 0, 0, 1, 0,], # A
            [ 0, 0, 0, 1, 0,], # O
            [ 0, 0, 0, 0, 1,], # B
        ]).float()
        self.I = torch.tensor([0, 1, 2, 3, 3, 4])

    def calc(self, preds, gts, batch):
        num_classes = preds.shape[-1]
        new_preds = torch.matmul(preds, self.mat[:num_classes, :].to(preds.device))
        new_gts = self.I.to(gts.device)[gts]
        return multi_accuracy(new_preds, new_gts)


class Trainer(BaseTrainer):
    def prepare(self):
        model = CompareModel(num_classes=self.config.num_classes(), base=self.config.base)
        if self.config.encoder == 'frozen':
            model.freeze_encoder()
        self.criterion = nn.CrossEntropyLoss()
        return model

    def create_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.config.lr)

    def create_scheduler(self):
        s = self.config.scheduler
        if s == 'static':
            return None
        m = re.match(r'^step_(\d+)$', s)
        if m:
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=int(m[1]), gamma=0.1)
        raise RuntimeError('Invalid scheduler')
        # return optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.1)

    def eval(self, inputs, gts, i):
        inputs = inputs.to(self.device)
        gts = gts.to(self.device)
        logits = self.model(inputs, activate=False)
        preds = torch.softmax(logits, dim=1)
        loss = self.criterion(logits, gts)
        return loss, logits.detach().cpu()

    def get_metrics(self):
        if self.config.code[:-1] == 'LMGAO':
            return {
                'acc5': MultiAccuracy(),
                'acc4': Acc4(),
                'acc3': Acc3(),
            }
        return {
            f'acc{len(self.config.unique_code())-1}': MultiAccuracy(),
        }

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
        if val_preds is None or val_gts is None:
            return
        self._visualize_confusion(ax, 'val', val_preds, val_gts)



class CLI(BaseMLCLI):
    class CommonArgs(BaseDLArgs):
        pass

    class TrainArgs(CommonArgs, TrainerConfig):
        # train param
        batch_size: int = Field(50, s='-B')
        num_workers: int = Field(4, s='-N')
        epoch: int = Field(10, s='-E')
        overwrite:bool = Field(False, s='-O')
        suffix: str = ''
        out: str = 'out/compare/{code}/fold{total_fold}_{fold}/{encoder}_{base}_{limit}{suffix}'

    def run_train(self, a:TrainArgs):
        config = TrainerConfig(**a.dict())

        dss = [
            FoldDataset(
                total_fold = a.total_fold,
                fold = a.fold,
                source = a.source,
                target = t,
                code = a.code,
                crop_size = a.crop_size,
                size = a.size,
                minimum_area = a.minimum_area,
                limit = a.limit,
                upsample = not config.noupsample,
                augmentation = t=='train',
                normalization = True,
                mean=MEAN,
                std=STD,
                # mean = 0.7,
                # std = 0.2,
            ) for t in ('train', 'test')
        ]

        out_dir = a.out.format(**a.dict())
        print('out dir:', out_dir)

        trainer = Trainer(
            config = config,
            out_dir = out_dir,
            train_dataset = dss[0],
            val_dataset = dss[1],
            use_gpu = not a.cpu,
            multi_gpu = True,
            overwrite = a.overwrite,
            experiment_name = a.source,
            fig_col_count = 2,
        )

        trainer.start(a.epoch)

    class ValidateArgs(BaseDLArgs):
        model_dir: str = Field(..., s='-d')
        target: str = Field('test', choices=['train', 'test', 'all'])
        batch_size: int = Field(16, s='-B')
        default: bool = False
        no_features: bool = False
        use_best: bool = False

    def run_validate(self, a:ValidateArgs):
        chp = 'checkpoint_best.pt' if a.use_best else 'checkpoint_last.pt'
        checkpoint = Checkpoint.from_file(J(a.model_dir, chp))
        config = TrainerConfig.from_file(J(a.model_dir, 'config.json'))
        model = TimmModel(name=config.model_name, num_classes=config.num_classes)
        print('config:', config)
        if config.nested in ['graph', 'hier']:
            model_state = {
                k.replace('model.', ''): v
                for k, v in checkpoint.model_state.items()
                if k.startswith('model.')
            }
        else:
            model_state = checkpoint.model_state
        model.load_state_dict(model_state)
        model = model.to(a.device()).eval()

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

        target = 'un' + a.target if a.default else a.target

        df = ds.df.copy()
        df[ds.unique_code] = -1.0
        df['pred'] = '_'

        num_chunks = math.ceil(len(df) / a.batch_size)
        tq = tqdm(range(num_chunks))

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
                    o = model(i, activate=True, with_feautres=False)
                else:
                    o, f = model(i, activate=True, with_feautres=True)
                    features = f.detach().cpu().numpy()
                    featuress.append(features)
                o = o.detach().cpu().numpy()
            df.loc[df.index[i0:i1], ds.unique_code] = o
            preds = pd.Series([ds.unique_code[i] for i in np.argmax(o, axis=1)], index=df.index[i0:i1])
            diags = df.loc[df.index[i0:i1], 'diag']

            df.loc[df.index[i0:i1], 'pred'] = preds

            map3 = {'L':'L', 'M':'M', 'G':'G', 'A':'G', 'O':'G', 'B':'B'}
            map4 = {'L':'L', 'M':'M', 'G':'G', 'A':'I', 'O':'I', 'B':'B'}

            correct = (diags == preds).astype(int)
            df.loc[df.index[i0:i1], 'correct'] = correct
            acc = correct.mean()
            message = f'Acc:{acc:.3f}'
            if config.code == 'LMGAOB':
                correct3 = (diags.map(map3) == preds.map(map3)).astype(int)
                correct4 = (diags.map(map4) == preds.map(map4)).astype(int)
                df.loc[df.index[i0:i1], 'correct3'] = correct3
                df.loc[df.index[i0:i1], 'correct4'] = correct4
                acc3 = correct3.mean()
                acc4 = correct4.mean()
                message += f' Acc3:{acc3:.3f} Acc4:{acc4:.3f}'
            tq.set_description(f'{i0} - {i1}: {message}')
            tq.refresh()

        df.to_excel(with_wrote(J(a.model_dir, f'validate_{target}.xlsx')))

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
            torch.save(data, J(a.model_dir, f'features_{target}.pt'))


    class PredictArgs(BaseDLArgs):
        model_dir: str = Field(..., s='-d')
        src: str = Field(..., s='-s')
        dest: str = ''
        cam: bool = Field(False, s='-c')
        cam_label: str = ''
        center: int = -1
        size: int = -1
        cols: int = 3
        use_last: bool = False

        plot: bool = False
        show: bool = False

    def run_predict(self, a:PredictArgs):
        checkpoint = Checkpoint.from_file(J(a.model_dir, 'checkpoint_last.pt'))
        config = TrainerConfig.from_file(J(a.model_dir, 'config.json'))
        model = TimmModel(name=config.model_name, num_classes=config.num_classes)
        model.load_state_dict(checkpoint.model_state)
        model = model.to(a.device()).eval()

        d = a.dest or ('cam' if a.cam else 'pred')
        dest_dir = J(with_mkdir(a.model_dir, d))
        print('dest:', dest_dir)

        gradcam = CAM.GradCAM(
            model=model,
            target_layers=model.get_cam_layers(),
            use_cuda=not a.cpu) if a.cam else None

        transform = transforms.Compose([v for v in [
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ] if v])

        if os.path.splitext(a.src)[-1] == '.xlsx':
            df_report = pd.read_excel(a.src, sheet_name='images')
            # caseごとの最初の画像と、間違えてえたものを評価
            selected_rows = []
            for name, rows in df_report.groupby('name'):
                # select first
                selected_rows.append(rows.iloc[:1])
            selected_rows.append(df_report[df_report['correct'] < 1])
            selected_rows = pd.concat(selected_rows)

            paths = []
            dests = []
            source = config.source.split('_')[0]
            for i, row in selected_rows.iterrows():
                paths.append(J('data/images', source, row['diag_org'], row['image_name']))
                dests.append(J(dest_dir, row['gt']))
            images = [Image.open(p) for p in paths]
        else:
            images, paths  = load_images_from_dir_or_file(a.src, with_path=True)
            dests = [dest_dir for _ in range(len(images))]

        imagess = []
        col_counts = []
        row_counts = []
        if a.size > 0:
            for image in images:
                ggg = grid_split(image, size=a.size,  overwrap=False, flattern=False)
                ii = []
                for gg in ggg:
                    for g in gg:
                        ii.append(g)
                col_counts.append(len(ggg[0]))
                row_counts.append(len(ggg))
                imagess.append(ii)
        else:
            crop = transforms.CenterCrop(a.center) if a.center > 0 else lambda x: x
            for image in images:
                imagess.append([crop(image)])
                col_counts.append(1)
                row_counts.append(1)

        rows = math.ceil(len(images) / a.cols)
        cols = min(a.cols, len(images))

        fig = plt.figure(figsize=(cols*6, rows*4))

        unique_code = config.unique_code()

        for idx, (
            path, images, dest, col_count, row_count
        ) in enumerate(zip(paths, imagess, dests, col_counts, row_counts)):

            oo = []
            # drews = [[None]*col_count]*row_count
            drews = [[None for _ in range(col_count)] for __ in range(row_count)]

            for i, image in enumerate(images):
                t = transform(image)[None, ...].to(a.device())
                with torch.set_grad_enabled(False):
                    o = model(t, activate=True)
                    o = o.detach().cpu().numpy()[0]
                oo.append(o)

                pred_id = np.argmax(o)
                pred = unique_code[pred_id]

                if a.cam:
                    cam_class_id = unique_code.index(a.cam_label) if a.cam_label else pred_id
                    cam_class = unique_code[cam_class_id]
                    targets = [ClassifierOutputTarget(cam_class_id)]

                    mask = gradcam(input_tensor=t, targets=targets)[0]
                    vis = show_cam_on_image(np.array(image)/255, mask, use_rgb=True)
                    # overwrite variable
                    image = Image.fromarray(vis)

                draw_frame(image, o, unique_code)
                drews[i // col_count][i % col_count] = image

            row_images = []
            for y, row in enumerate(drews):
                row_image_list = []
                for x, d in enumerate(row):
                    row_image_list.append(np.array(d))
                h = cv2.hconcat(row_image_list)
                row_images.append(h)
            merged_image = Image.fromarray(cv2.vconcat(row_images))

            o = np.stack(oo).mean(axis=0)

            pred_id = np.argmax(o)
            pred = unique_code[pred_id]
            label = ' '.join([f'{c}:{int(v*100):3d}' for c, v in zip(unique_code, o)])
            h, w = t.shape[2], t.shape[3]
            print(f'{path}: ({w}x{h}) {pred} ({label})')

            name = os.path.splitext(os.path.basename(path))[0]
            merged_image.save(J(with_mkdir(dest), f'{name}.jpg'))

            merged_image.close()

            if not a.plot:
                continue
            ax = fig.add_subplot(rows, cols, idx+1)
            ax.imshow(merged_image)
            title = f'{name} {pred} ({label})'
            if a.cam_label:
                title += f' CAM: {cam_class}'
            ax.set_title(title)
            ax.set(xlabel=None, ylabel=None)

        if a.plot:
            plt.savefig()
            if a.show:
                plt.show()


if __name__ == '__main__':
    cli = CLI()
    cli.run()
