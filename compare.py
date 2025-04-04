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
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt, cm as colormap
import matplotlib
import seaborn as sns
from sklearn import metrics as skmetrics
from sklearn.preprocessing import label_binarize
from scipy import stats
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
from endaaman.ml.utils import hover_images_on_scatters

from models import CompareModel
from datasets import FoldDataset, MEAN, STD
from datasets.ebrains import EBRAINSDataset, get_ebrains_df
from utils import draw_frame, pad16, grid_compose_image


torch.multiprocessing.set_sharing_strategy('file_system')
np.set_printoptions(suppress=True, floatmode='fixed')
J = os.path.join

COLORS = {
    'L': '#1f77b4',
    'M': '#ff7f0e',
    'G': '#2ca02c',
    'A': '#d62728',
    'O': '#9467bd',
    # 'B': '#ac64ad',
    'B': '#8c564b',
}

FG_COLORS = {
    'L': 'white',
    'M': 'white',
    'G': 'white',
    'A': 'white',
    'O': 'white',
    'B': 'white',
}


def find_closest_pair(x, r):
    closest_diff = float('inf')
    closest_a = None
    closest_b = None
    for a in range(1, x + 1):
        if x % a != 0:
            continue
        b = x // a
        current_r = a / b
        diff = abs(current_r - r)
        if diff < closest_diff:
            closest_diff = diff
            closest_a = a
            closest_b = b
    return closest_a, closest_b


def get_reshaper(name, width, height):
    if 'cnn' in name:
        return None

    feature_size = 1024

    def reshape_transform(tensor):
        tensor = tensor[:, 1:, :]
        w, h = find_closest_pair(tensor.numel()//feature_size, width/height)
        result = tensor.reshape(
            tensor.size(0),
            h,
            w,
            tensor.size(-1)
        )
        # ABCD -> ABDC -> ADBC
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    return reshape_transform




class TrainerConfig(BaseTrainerConfig):
    # model
    base: str = 'uni'
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
    lr_scaling_factor: float = 1
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
            # L, M, G, I, B
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
        model = CompareModel(
                base=self.config.base,
                num_classes=self.config.num_classes(),
                frozen=self.config.encoder == 'frozen')
        self.criterion = nn.CrossEntropyLoss()
        return model

    def create_optimizer(self):
        if self.config.lr_scaling_factor == 1.0:
            # Backward compat
            # return optim.Adam([
            #     {'params': self.model.base.parameters(), 'lr': self.config.lr},
            #     {'params': self.model.fc.parameters(), 'lr': self.config.lr},
            # ])
            return optim.Adam(self.model.parameters(), lr=self.config.lr)
        params = [
            {'params': self.model.base.parameters(), 'lr': self.config.lr * self.config.lr_scaling_factor},
            {'params': self.model.fc.parameters(), 'lr': self.config.lr},
        ]
        return optim.Adam(params)

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
            device = a.device,
            multi_gpu = True,
            overwrite = a.overwrite,
            experiment_name = a.source,
            fig_col_count = 2,
            save_best = False,
        )

        trainer.start(a.epoch)

    class ValidateArgs(BaseDLArgs):
        model_dir: str = Field(..., s='-d')
        target: str = Field('test', choices=['train', 'test', 'all'])
        batch_size: int = Field(100, s='-B')
        use_best: bool = False
        with_features: bool = False
        skip_checkpoint: bool = False
        limit: int = -1

    def run_validate(self, a:ValidateArgs):
        config = TrainerConfig.from_file(J(a.model_dir, 'config.json'))
        print('config:', config)

        dest_path = J(a.model_dir, f'validate_{a.target}.xlsx')
        feature_path = J(a.model_dir, f'features_{a.target}.pt')
        if os.path.exists(dest_path):
            if a.with_features:
                if not feature_path:
                    print(f'Skipping: {dest_path} and {feature_path} exist.')
                    return
            else:
                print(f'Skipping: {dest_path} exists.')
                return

        model = CompareModel(num_classes=config.num_classes(),
                             frozen=config.encoder == 'frozen',
                             base=config.base)
        if not a.skip_checkpoint:
            chp = 'checkpoint_best.pt' if a.use_best else 'checkpoint_last.pt'
            checkpoint = Checkpoint.from_file(J(a.model_dir, chp))
            model.load_state_dict(checkpoint.model_state)
        model = model.to(a.device).eval()

        transform = transforms.Compose([
            transforms.CenterCrop(config.crop_size),
            transforms.Resize(config.size),
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
                limit=a.limit,
        )

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
                image = ds.load_from_row(row)
                tt.append(transform(image))
                image.close()

            tt = torch.stack(tt)
            with torch.set_grad_enabled(False):
                i = tt.to(a.device)
                if a.with_features:
                    o, f = model(i, activate=True, with_features=True)
                    features = f.detach().cpu().numpy()
                    featuress.append(features)
                else:
                    o = model(i, activate=True, with_features=False)
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

        df.to_excel(with_wrote(dest_path))

        if a.with_features:
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
            torch.save(data, feature_path)


    class EbrainsArgs(BaseDLArgs):
        model_dir: str = Field(..., s='-d')
        batch_size: int = Field(100, s='-B')
        use_best: bool = False
        with_features: bool = False
        skip_evaluation: bool = False
        count: int = -1

    def run_ebrains(self, a:EbrainsArgs):
        map3 = {'L':'L', 'M':'M', 'G':'G', 'A':'G', 'O':'G', 'B':'B'}

        config = TrainerConfig.from_file(J(a.model_dir, 'config.json'))
        print('config:', config)

        unique_code = config.unique_code()

        if a.skip_evaluation:
            df_patches = pd.read_excel(J(a.model_dir, 'ebrains.xlsx'), sheet_name='patches')
        else:
            model = CompareModel(num_classes=config.num_classes(),
                                 frozen=config.encoder == 'frozen',
                                 base=config.base)
            checkpoint = Checkpoint.from_file(J(a.model_dir, 'checkpoint_last.pt'))
            model.load_state_dict(checkpoint.model_state)
            model = model.to(a.device).eval()

            ds = EBRAINSDataset(crop_size=config.crop_size, patch_size=config.size, code=config.code)
            loader = DataLoader(ds, a.batch_size, shuffle=False)
            results = []
            preds = []
            featuress = []

            print('Evaluating')
            for i, (xx, gts, idxs) in tqdm(enumerate(loader), total=len(loader)):
                items = [ds.items[i] for i in idxs]
                with torch.set_grad_enabled(False):
                    if a.with_features:
                        yy, features = model(xx.to(a.device), activate=True, with_features=True)
                        yy = yy.detach().cpu()
                        features = features.detach().cpu().numpy()
                        featuress.append(features)
                    else:
                        yy = model(xx.to(a.device), activate=True).cpu().detach()

                preds = torch.argmax(yy, dim=1)
                for item, y, gt, pred in zip(items, yy, gts, preds):
                    values = dict(zip([*config.code], y.tolist()))
                    pred_label = unique_code[pred]
                    r = {
                        **item.asdict(),
                        'pred': pred_label,
                        'correct': int(pred == gt),
                        'correct3': int(map3[item.label] == map3[pred_label]),
                        **values,
                    }
                    del r['image']
                    results.append(r)

                if (a.count > 0) and (i >= a.count):
                    break
            df_patches = pd.DataFrame(results)

            if a.with_features:
                feature_path = J(a.model_dir, 'features_ebrains.pt')
                features = np.concatenate(featuress)
                features = features.reshape(features.shape[0], features.shape[1])
                data = [
                    dict(zip(['name', 'path', 'diag_org', 'pred', 'feature'], values))
                    for values in zip(
                        df_patches['name'],
                        df_patches['path'],
                        df_patches['label'],
                        df_patches['pred'],
                        features
                    )
                ]
                torch.save(data, with_wrote(feature_path))

        results_cases = []
        for name, rows in df_patches.groupby('name'):
            assert rows['label'].nunique() == 1
            pred = []
            for c in unique_code:
                pred.append(rows[c].mean())
            pred_idx = np.argmax(pred)
            # if B is major, yield pred from second choice
            if unique_code[pred_idx] == 'B':
                pred_idx = np.argsort(pred)[-2]
            pred_label = unique_code[pred_idx]

            gt_label = rows.iloc[0]['label']
            results_cases.append({
                'name': name,
                'label': gt_label,
                'pred': pred_label,
                'count': len(rows),
                'correct': int(gt_label == pred_label),
                'correct3': int(map3[gt_label] == map3[pred_label]),
                **dict(zip(unique_code, pred))
            })
        df_cases = pd.DataFrame(results_cases)

        y_true, y_pred = df_cases['label'], df_cases['pred']
        patch_accuracy = df_patches['correct'].mean()
        report = skmetrics.classification_report(y_true, y_pred, zero_division=1.0, output_dict=True)
        report['patch acc'] = patch_accuracy
        df_report = pd.DataFrame(report).T

        y_true3, y_pred3 = y_true.map(map3), y_pred.map(map3)
        patch_accuracy3 = df_patches['correct3'].mean()
        report3 = skmetrics.classification_report(y_true3, y_pred3, zero_division=1.0, output_dict=True)
        report3['patch acc'] = patch_accuracy3
        df_report3 = pd.DataFrame(report3).T

        with pd.ExcelWriter(with_wrote(J(a.model_dir, 'ebrains.xlsx'))) as w:
            df_patches.to_excel(w, sheet_name='patches', index=False)
            df_cases.to_excel(w, sheet_name='cases', index=False)
            df_report.to_excel(w, sheet_name='report', index=True)
            df_report3.to_excel(w, sheet_name='report3', index=True)



    class DrawSamplesArgs(CommonArgs):
        model_dir: str = Field(..., s='-d')
        batch_size: int = Field(500, s='-B')
        target: str = Field('test', choices=['train', 'test', 'ebrains'])
        diag: list[str] = Field(list('LMGAO'), choices=list('LMGAOB'))
        names: list[str] = Field([])
        order: int = 1
        nocrop: bool = False

    def run_draw_samples(self, a:DrawSamplesArgs):
        m = re.match(r'^.*fold5_(\d).*$', a.model_dir)
        if not m:
            print('Invalid dir', a.model_dir)
            return
        fold = int(m[1])

        checkpoint = Checkpoint.from_file(J(a.model_dir, 'checkpoint_last.pt'))
        config = TrainerConfig.from_file(J(a.model_dir, 'config.json'))

        model = CompareModel(num_classes=config.num_classes(),
                             frozen=config.encoder == 'frozen',
                             base=config.base)
        model.load_state_dict(checkpoint.model_state)
        model = model.to(a.device).eval()
        model.unfreeze_encoder()

        items = []

        if a.target in ['train', 'test']:
            df = pd.read_excel('data/tiles/enda4_512/folds5.xlsx')
            if a.target == 'test':
                df = df[df['fold'] == config.fold]
            else:
                df = df[df['fold'] != config.fold]
            for i, row in df.iterrows():
                name, diag = row['name'], row['diag']
                items.append({
                    'diag': diag,
                    'name': name,
                    'path': f'data/images/enda4/{diag}/{name}_{a.order:02d}.jpg',
                })
        elif a.target == 'ebrains':
            for d in sorted(os.listdir('data/EBRAINS/')):
                m = re.search(r'[LMGAO]', d)
                if m is None:
                    print('Skip', f'data/EBRAINS/{d}')
                    continue
                diag = m.group()
                for filename in sorted(os.listdir(f'data/EBRAINS/{d}/')):
                    name, ext = os.path.splitext(filename)
                    if ext != '.jpg':
                        continue
                    items.append({
                        'diag': diag,
                        'name': name,
                        'path': f'data/EBRAINS/{d}/{filename}',
                    })
        else:
            raise RuntimeError(f'Invalid target', a.target)

        unique_code = config.unique_code()

        transform = transforms.Compose([
            # transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

        items = [i for i in items if i['diag'] in a.diag]

        if len(a.names) > 0:
            items = [i for i in items if i['name'] in a.names]

        if len(items) == 0:
            print('No items are remained.')
            return

        tq = tqdm(items, position=0)
        for item in tq:
            name, diag = item['name'], item['diag']
            orig_image = Image.open(item['path'])
            if not a.nocrop:
                crop_width = (orig_image.width//512)*512
                crop_height = (orig_image.height//512)*512
                orig_image = orig_image.crop((0, 0, crop_width, crop_height))
            ggg = grid_split(orig_image, 512, overwrap=False, flattern=False)
            H = len(ggg)
            W = len(ggg[0])
            scale = 224/512
            # scale = 1
            gt_index = unique_code.index(diag)
            ttt = []
            tq2 = tqdm(total=H*W, leave=False, position=1)
            for y, gg in enumerate(ggg):
                tt = []
                for x, g in enumerate(gg):
                    tile_orig = g.convert('RGBA')
                    if not a.nocrop:
                        tile_resized, pos = pad16(g.resize((round(g.width*scale), round(g.height*scale))), with_dimension=True)
                    else:
                        tile_resized = g.resize((224, 224))
                        pos = (0, 0)
                    gradcam = CAM.GradCAMPlusPlus(
                    # gradcam = CAM.GradCAM(
                        model=model,
                        target_layers=model.get_cam_layers(),
                        reshape_transform=get_reshaper(config.base, tile_resized.width, tile_resized.height),
                    )

                    t = transform(tile_resized)[None, ...]
                    with torch.no_grad():
                        pred = model(t.to(a.device), activate=True).cpu().detach()[0].numpy()
                    pred_index = np.argmax(pred)
                    pred_label = unique_code[pred_index]
                    pred_dict = {k: pred[i] for i, k in enumerate(unique_code) }
                    text = ' '.join([f'{k}:{round(pred_dict[k]*100)}' for k in 'GAOLMB'])
                    frame = draw_frame(g.size, text, COLORS[pred_label], FG_COLORS[pred_label])

                    # print(pred_index, pred_label, pred)
                    targets =  [ClassifierOutputTarget(pred_index)]
                    # targets =  [ClassifierOutputTarget(gt_index)]

                    cam_mask = gradcam(
                        input_tensor=t,
                        targets=targets,
                        eigen_smooth=True
                    )[0]
                    # mask = Image.fromarray((grayscale_cam * 255).astype(np.uint8))
                    cam_base = Image.fromarray(np.uint8(colormap.jet(cam_mask)*255))

                    cam_resized = cam_base.crop((pos[0], pos[1], tile_resized.width, tile_resized.height))
                    cam_orig = cam_resized.resize(tile_orig.size)
                    tile_orig = Image.blend(tile_orig, cam_orig, alpha=0.5)
                    tile_orig.paste(frame, (0, 0, frame.width, frame.height), mask=frame)
                    tt.append(tile_orig)
                    tq2.update(1)
                ttt.append(tt)

            dir_suffix = '_nocrop' if a.nocrop else '_crop'

            grid_image = grid_compose_image(ttt)
            grid_path = J(a.model_dir, f'pred{dir_suffix}', a.target, diag, f'{name}.jpg')
            os.makedirs(os.path.dirname(grid_path), exist_ok=True)
            grid_image.convert('RGB').save(grid_path)

            orig_path = J(f'data/tmp/cam{dir_suffix}', f'{fold}_{a.target}', diag, f'{name}.jpg')
            os.makedirs(os.path.dirname(orig_path), exist_ok=True)
            # if not os.path.exists(orig_path):
            #     orig_image.save(orig_path)
            orig_image.save(orig_path)

            tq.set_description(grid_path)



    class CalcResultsArgs(CommonArgs):
        base: str = 'uni'
        encoder: str = Field('frozen', choices=['frozen', 'unfrozen'])
        limit: int = Field(100, choices=[10, 25, 50, 100, 500])
        fold: int = Field(-1, choices=[-1, 0, 1, 2, 3, 4])
        target: str = Field('test', s='-t')
        overwrite: bool = Field(False, s='-O')
        suffix: str = Field('', s='-S')

    def run_calc_results(self, a):
        unique_code = list('LMGAOB')
        map3 = {'L':'L', 'M':'M', 'G':'G', 'A':'G', 'O':'G', 'B':'B'}
        map4 = {'L':'L', 'M':'M', 'G':'G', 'A':'I', 'O':'I', 'B':'B'}

        if a.fold < 0:
            dest_dir= f'out/compare/results/{a.encoder}_{a.base}_{a.limit}{a.suffix}'
            dfs = []
            for fold in range(5):
                model_dir = f'out/compare/LMGAOB/fold5_{fold}/{a.encoder}_{a.base}_{a.limit}/'
                if a.suffix:
                    model_dir += a.suffix
                excel_path = J(model_dir, f'validate_{a.target}.xlsx')
                dfs.append(pd.read_excel(excel_path))
                print('loaded', excel_path)
            df_patches = pd.concat(dfs)
        else:
            model_dir = f'out/compare/LMGAOB/fold5_{a.fold}/{a.encoder}_{a.base}_{a.limit}'
            if a.suffix:
                model_dir += a.suffix
            dest_dir= J(model_dir, a.target)
            df_patches = pd.read_excel(J(model_dir, f'validate_{a.target}.xlsx'))

        output_path = J(dest_dir, 'report.xlsx')
        if os.path.exists(output_path):
            if a.overwrite:
                print('Overwriting', output_path)
            else:
                print('Skipping, output_path exists:', output_path)
                return

        os.makedirs(dest_dir, exist_ok=True)
        print('DEST', dest_dir)

        data_by_case = []

        for name, items in tqdm(df_patches.groupby('name')):
            diag_org, diag = items.iloc[0][['diag_org', 'diag']]
            if diag != 'B':
                items = items[items['area'] > 0.6]

            preds = items[unique_code]

            preds_sum = np.sum(preds, axis=0)
            preds_sum = preds_sum / np.sum(preds_sum)
            pred_sum = unique_code[np.argmax(preds_sum)]

            preds_label = np.argmax(preds, axis=1)

            gt_label = unique_code.index(diag)
            acc = np.mean(preds_label == gt_label)

            unique_values, counts = np.unique(preds_label, return_counts=True)
            pred_vote = unique_code[unique_values[np.argmax(counts)]]

            d = {
                'name': name,
                'diag_org': diag_org,
                'gt': diag,
                'pred(vote)': pred_vote,
                'pred(sum)': pred_sum,
                'correct': int(diag == pred_sum),
                'acc': acc,
            }
            d['correct3'] = int(map3[diag_org] == map3[pred_sum])
            d['correct4'] = int(map4[diag_org] == map4[pred_sum])
            for p, code in zip(preds_sum, unique_code):
                d[code] = p
            data_by_case.append(d)

        df = pd.DataFrame(data_by_case).sort_values(['diag_org'])

        print('Acc3 by case', df['correct3'].mean())
        print('Acc4 by case', df['correct4'].mean())

        y_true = df['gt']
        y_pred = df['pred(sum)']

        y_score = df[unique_code].values
        # One-hot encoding
        y_true_bin = label_binarize(y_true, classes=unique_code)

        ##* ROC
        print('ROC AUC')

        fpr, tpr, roc_auc = {}, {}, {}
        for code in unique_code:
            p = df[code]
            gt = df['gt'] == code
            fpr[code], tpr[code], __t = skmetrics.roc_curve(gt, p)
            roc_auc[code] = skmetrics.auc(fpr[code], tpr[code])
            plt.plot(fpr[code], tpr[code], label=f'{code}: AUC={roc_auc[code]:.3f}')
            plt.legend()
            plt.savefig(J(dest_dir, f'{code}.png'))
            plt.close()
            print(code, roc_auc[code])

        fpr['micro'], tpr['micro'], _ = skmetrics.roc_curve(y_true_bin.ravel(), y_score.ravel())
        roc_auc['micro'] = skmetrics.auc(fpr['micro'], tpr['micro'])

        all_fpr = np.unique(np.concatenate([fpr[code] for code in unique_code]))
        mean_tpr = np.zeros_like(all_fpr)
        for code in unique_code:
            mean_tpr += np.interp(all_fpr, fpr[code], tpr[code])
        mean_tpr /= len(unique_code)
        fpr['macro'] = all_fpr
        tpr['macro'] = mean_tpr
        roc_auc['macro'] = skmetrics.auc(fpr['macro'], tpr['macro'])
        print('micro', roc_auc['micro'])
        print('macro', roc_auc['macro'])
        plt.plot(fpr['macro'], tpr['macro'], label=f'AUC={roc_auc["macro"]:.3f}')
        plt.legend()
        plt.savefig(J(dest_dir, 'roc.png'))
        plt.close()

        ##* Reports
        report = skmetrics.classification_report(y_true, y_pred, output_dict=True)
        print(f'Classification Report by:')
        print(report)
        df_report = pd.DataFrame(report)
        df_report['auc'] = roc_auc['macro']
        df_report['patch acc'] = np.mean(df_patches['correct'])
        df_report = df_report.T
        print(df_report)

        y_true3, y_pred3 = y_true.map(map3), y_pred.map(map3)
        report3 = skmetrics.classification_report(y_true3, y_pred3, output_dict=True)
        report3['patch acc'] = np.mean(df_patches['correct3'])
        df_report3 = pd.DataFrame(report3).T

        # report4 = skmetrics.classification_report(y_true.map(map4), y_pred.map(map4), output_dict=True)
        # df_report4 = pd.DataFrame(report4).T

        with pd.ExcelWriter(with_wrote(output_path), engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='cases', index=False)
            df_report.to_excel(writer, sheet_name='report')
            df_report3.to_excel(writer, sheet_name='report3')
            # df_report4.to_excel(writer, sheet_name='report4')

    class SummaryCvArgs(CommonArgs):
        coarse: bool = False

    def run_summary_cv(self, a):
        conds = [
            (
                'frozen_gigapath_{}', 'Prov-GigaPath(LP)',
                [10, 25, 100, 500],
            ), (
                'unfrozen_uni_{}', 'UNI(FT)',
                [10, 25, 100, 500],
            ), (
                'frozen_uni_{}', 'UNI(LP)',
                [10, 25, 100, 500],
            ), (
                'unfrozen_ctranspath_{}', 'CTransPath(FT)',
                [10, 25, 100, 500],
            ), (
                'frozen_ctranspath_{}', 'CTransPath(LP)',
                [10, 25, 100, 500],
            ), (
                'unfrozen_baseline-vit_{}', r'ViT-L$\mathrm{_{IN}}$(FT)',
                [10, 25, 100, 500],
            ), (
                'frozen_baseline-vit_{}', r'ViT-L$\mathrm{_{IN}}$(LP)',
                [10, 25, 100, 500],
            ), (
                'unfrozen_random-vit_{}', r'ViT-L(RI)',
                [10, 25, 100, 500],
            ),(
                'unfrozen_baseline-cnn_{}', r'ResNet-RS 50$\mathrm{_{IN}}$(FT)',
                [10, 25, 100, 500],
            ), (
                'frozen_baseline-cnn_{}', r'ResNet-RS 50$\mathrm{_{IN}}$(LP)',
                [10, 25, 100, 500],
            ),
        ]

        metrics_fns = {
            'Patch Acc.': lambda df: df.loc['patch acc'].iloc[0],
            'Macro Acc.': lambda df: df.loc['accuracy'].iloc[0],
            'Macro Recall': lambda df: df.loc['macro avg', 'recall'],
            'Macro Prec.': lambda df: df.loc['macro avg', 'precision'],
            'Macro F1': lambda df: df.loc['macro avg', 'f1-score'],
            # 'AUROC ': lambda df: df[df.index == 'auc'].iloc[0, 0],
            'G Recall': lambda df: df.loc['G', 'recall'],
            'A Recall': lambda df: df.loc['A', 'recall'],
            'O Recall': lambda df: df.loc['O', 'recall'],
            'M Recall': lambda df: df.loc['M', 'recall'],
            'L Recall': lambda df: df.loc['L', 'recall'],

            'G Prec.': lambda df: df.loc['G', 'precision'],
            'A Prec.': lambda df: df.loc['A', 'precision'],
            'O Prec.': lambda df: df.loc['O', 'precision'],
            'M Prec.': lambda df: df.loc['M', 'precision'],
            'L Prec.': lambda df: df.loc['L', 'precision'],
        }

        skip_when_coarse = ['A Recall', 'O Recall', 'A Prec.', 'O Prec.']

        dfs_to_save = []

        for (cond_base, label, limits) in conds:
            dfs = []
            for limit in limits:
                cond = cond_base.format(limit)
                mm = []
                for fold in range(5):
                    p = f'out/compare/LMGAOB/fold5_{fold}/{cond}/test/report.xlsx'
                    df = pd.read_excel(p, 'report3' if a.coarse else 'report', index_col=0)
                    m = {
                        'fold': fold,
                        'cond': cond,
                        'limit': limit,
                        'label': label
                    }
                    for key, fn in metrics_fns.items():
                        if a.coarse and key in skip_when_coarse:
                            continue
                        try:
                            value = fn(df)
                        except Exception as e:
                            print('Error in', p)
                            print(df)
                            raise e
                        m[key] = fn(df)
                    mm.append(m)

                df = pd.DataFrame(mm)
                dfs.append(df)
            df = pd.concat(dfs).reset_index(drop=True)
            dfs_to_save.append(df)

        df_to_save = pd.concat(dfs_to_save).reset_index(drop=True)
        grains = 'coarse' if a.coarse else 'fine'
        with pd.ExcelWriter(with_wrote(f'out/figs/tables/results_{grains}_cv.xlsx')) as w:
            for limit, df in df_to_save.groupby('limit'):
                df.to_excel(w, sheet_name=f'{limit}')


    class CmArgs(CommonArgs):
        target = Field('cv', choices=['cv', 'ebrains'])
        base: str = 'uni'
        encoder: str = Field('frozen', choices=['frozen', 'unfrozen'])
        coarse: bool = False
        limit: int = 500
        with_b: bool= False
        subtype: bool = False
        noshow: bool = False

    def run_cm(self, a):
        if a.target != 'ebrains':
            gt_key = 'diag_org'
            pred_key = 'pred(sum)'
            base_path = 'out/compare/LMGAOB/fold5_{fold}/{cond}/test/report.xlsx'
            target_label = 'Local'
        else:
            gt_key = 'label'
            pred_key = 'pred'
            base_path = 'out/compare/LMGAOB/fold5_{fold}/{cond}/ebrains.xlsx'
            target_label = 'EBRAINS'

        unique_code = list('LMGAOB')

        mm = []
        for fold in range(5):
            cond = f'{a.encoder}_{a.base}_{a.limit}'
            # p = f'out/compare/LMGAOB/fold5_{fold}/{a.encoder}_{a.base}_{a.limit}/ebrains.xlsx'
            p = base_path.format(
                fold=fold,
                cond=cond,
            )
            df = pd.read_excel(p, 'cases', index_col=0)
            # drop B
            df = df[df[gt_key] != 'B']
            m = df[[gt_key]].rename(columns={gt_key: 'gt'}).copy()
            y_pred = []
            for i, row in df.iterrows():
                p = row[pred_key]
                if not a.with_b and p == 'B':
                    print('pred B', row[unique_code])
                    p = unique_code[row[unique_code].argsort().iloc[-2]]
                    print('select second', p)
                y_pred.append(p)
            m['pred'] = y_pred
            mm.append(m)
        labels = list('GAOML')
        if a.with_b:
            labels += ['B']
        df = pd.concat(mm)

        if a.target == 'ebrains':
            # if ebrains do ensemble
            df_map = get_ebrains_df()
            new_data = []
            for name, rows in df.groupby('name'):
                subtype = df_map.loc[name, 'subtype']
                # if a.drop_transitional and (subtype in ['3. G_AA-IDH-wild', '3. G_DA-IDH-wild', '4. A_GBM-IDH-mut']):
                #     continue
                counts = rows['pred'].value_counts()
                top_pred = counts[counts == counts.max()]
                if len(top_pred) == 1:
                    pred = counts.index[0]
                else:
                    # 同率一位
                    pred =top_pred.index[0]
                new_data.append({
                    'name': name,
                    'subtype': subtype,
                    'gt': rows.iloc[0]['gt'],
                    'pred': pred,
                })
            df = pd.DataFrame(new_data)

        cond = f'{a.encoder}_{a.base}'
        label = {
            'frozen_uni': 'UNI(LP)',
            'unfrozen_uni': 'UNI(FT)',
            'frozen_baseline-vit': r'ViT-L$\mathrm{_{IN}}$(LP)',
            'unfrozen_baseline-vit': r'ViT-L$\mathrm{_{IN}}$(FT)',
        }.get(cond, cond)
        label = f'{label} - {target_label}'

        plt.rcParams.update({
            # 'font.size': 12,
            # 'axes.titlesize': 14,
            # 'axes.labelsize': 12,
            # 'xtick.labelsize': 12,
            # 'ytick.labelsize': 10,
            # 'legend.fontsize': 16,
            'figure.dpi': 300,
        })

        if a.target == 'ebrains' and a.subtype:
            subtypes = [
                'GBM, IDH(-)',
                'AA, IDH(-)',
                'DA, IDH(-)',
                'AA, IDH(+)',
                'DA, IDH(+)',
                'GBM, IDH(+)',
                'AO',
                'O',
                'M',
                'L',
            ]
            cm = pd.DataFrame(0, index=labels, columns=subtypes)
            for _, row in df.iterrows():
                cm.loc[row['pred'], row['subtype']] += 1
            gt_labels = subtypes
        else:
            cm = skmetrics.confusion_matrix(df['pred'], df['gt'], labels=labels)
            gt_labels = labels

        plt.figure(figsize=(6, 4) if a.target == 'ebrains' and a.subtype else (4, 4))
        heatmap = sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=gt_labels,
                yticklabels=labels,
                square=True)
        plt.xlabel('Groud truth')
        plt.ylabel('Predicted')
        plt.title(label)
        plt.yticks(rotation=0)
        if a.target == 'ebrains' and a.subtype:
            plt.xticks(rotation=45, ha='right')

        report = skmetrics.classification_report(df['gt'], df['pred'])
        print(report)
        plt.tight_layout()
        # plt.subplots_adjust(bottom=0.4)
        grains = 'coarse' if a.coarse else 'fine'
        suffix = '_subtype' if a.subtype else ''
        plt.savefig(with_wrote(J('out/figs/cm', f'cm_{a.target}_{grains}_{a.encoder}_{a.base}{suffix}.png'), True), dpi=300)
        if not a.noshow:
            plt.show()

    class SummaryEbrainsArgs(CommonArgs):
        coarse: bool = False

    def run_summary_ebrains(self, a):
        conds = [
            (
                'frozen_gigapath_{}', 'Prov-GigaPath(LP)',
                [10, 25, 100, 500],
            ), (
                'unfrozen_uni_{}', 'UNI(FT)',
                [10, 25, 100, 500],
            ), (
                'frozen_uni_{}', 'UNI(LP)',
                [10, 25, 100, 500],
            ), (
                'unfrozen_ctranspath_{}', 'CTransPath(FT)',
                [10, 25, 100, 500],
            ), (
                'frozen_ctranspath_{}', 'CTransPath(LP)',
                [10, 25, 100, 500],
            ), (
                'unfrozen_baseline-vit_{}', r'ViT-L$\mathrm{_{IN}}$(FT)',
                [10, 25, 100, 500],
            ), (
                'frozen_baseline-vit_{}', r'ViT-L$\mathrm{_{IN}}$(LP)',
                [10, 25, 100, 500],
            ), (
                'unfrozen_random-vit_{}', r'ViT-L(RI)',
                [10, 25, 100, 500],
            ),(
                'unfrozen_baseline-cnn_{}', r'ResNet-RS 50$\mathrm{_{IN}}$(FT)',
                [10, 25, 100, 500],
            ), (
                'frozen_baseline-cnn_{}', r'ResNet-RS 50$\mathrm{_{IN}}$(LP)',
                [10, 25, 100, 500],
            ),
        ]

        metrics_fns = {
            'Patch Acc.': lambda df: df.loc['patch acc'].iloc[0],
            'Macro Acc.': lambda df: df.loc['accuracy'].iloc[0],
            'Macro Recall': lambda df: df.loc['macro avg', 'recall'],
            'Macro Prec.': lambda df: df.loc['macro avg', 'precision'],
            'Macro F1': lambda df: df.loc['macro avg', 'f1-score'],
            # 'AUROC ': lambda df: df[df.index == 'auc'].iloc[0, 0],
            'G Recall': lambda df: df.loc['G', 'recall'],
            'A Recall': lambda df: df.loc['A', 'recall'],
            'O Recall': lambda df: df.loc['O', 'recall'],
            'M Recall': lambda df: df.loc['M', 'recall'],
            'L Recall': lambda df: df.loc['L', 'recall'],

            'G Prec.': lambda df: df.loc['G', 'precision'],
            'A Prec.': lambda df: df.loc['A', 'precision'],
            'O Prec.': lambda df: df.loc['O', 'precision'],
            'M Prec.': lambda df: df.loc['M', 'precision'],
            'L Prec.': lambda df: df.loc['L', 'precision'],
        }
        skip_when_coarse = ['A Recall', 'O Recall', 'A Prec.', 'O Prec.']
        target_sheet_name = 'report3' if a.coarse else 'report'

        results = []
        for cond_base, label, limits in conds:
            for limit in limits:
                cond = cond_base.format(limit)
                for fold in range(5):
                    df_path = f'out/compare/LMGAOB/fold5_{fold}/{cond}/ebrains.xlsx'
                    if not os.path.exists(df_path):
                        raise RuntimeError('Invalid path', df_path)
                    df = pd.read_excel(df_path, sheet_name=target_sheet_name, index_col=0)
                    r = {
                        'fold': fold,
                        'cond': cond,
                        'limit': limit,
                        'label': label,
                    }
                    for key, fn in metrics_fns.items():
                        if a.coarse and key in skip_when_coarse:
                            continue
                        r[key] = fn(df)
                    results.append(r)

        df_results = pd.DataFrame(results)

        grains = 'coarse' if a.coarse else 'fine'
        with pd.ExcelWriter(with_wrote(f'out/figs/tables/results_{grains}_ebrains.xlsx')) as w:
            for limit, df in df_results.groupby('limit'):
                df.to_excel(w, sheet_name=f'{limit}')


    class PreClusterArgs(CommonArgs):
        model_dir: str = Field(..., s='-d')
        train_file: str = 'features_train.pt'
        val_file: str = 'features_test.pt'
        eb_file: str = 'features_ebrains.pt'
        train_count: int = 10
        val_count: int = 10
        eb_count: int = 5
        n_neighbors: int = 70
        min_dist: float = 0.5
        spread: float = 1.0
        with_suffix: bool = False
        show: bool = False
        hover: bool = False

    def run_pre_cluster(self, a):
        from umap import UMAP
        # matplotlib.use('QtAgg')

        df_meta_origins = pd.read_excel('data/meta_origin.xlsx')
        df_meta_origins['origin'] = df_meta_origins['origin'].str.capitalize()

        val_data = torch.load(J(a.model_dir, a.val_file))
        eb_data = torch.load(J(a.model_dir, a.eb_file))

        cols = ['name', 'diag_org', 'pred', 'feature', 'path']

        df = pd.DataFrame(val_data)
        df['Dataset'] = 'Local(Val)'
        train_data_path = J(a.model_dir, a.train_file)
        if os.path.exists(train_data_path):
            train_data = torch.load(train_data_path)
            df_train = pd.DataFrame(train_data)
            df_train['Dataset'] = 'Local(Train)'
            df = pd.concat([df, df_train])
        else:
            print(f'info: train data does not exist {train_data_path}')

        df['path'] = [
            os.path.abspath(f'cache/tiles/enda4_512/{r.diag_org}/{r['name']}/{r.filename}')
            for i, r in df.iterrows()
        ]
        df = df[cols + ['Dataset']]
        df = pd.merge(df, df_meta_origins, on='name', how='left')
        df.fillna('', inplace=True)

        df_ebrains = pd.DataFrame(eb_data)
        df_ebrains = df_ebrains[cols]
        df_ebrains['Dataset'] = 'EBRAINS'
        df_ebrains['origin'] = ''
        df_ebrains.loc[df_ebrains['diag_org'] == 'M', 'origin'] = 'Meta(EBRAINS)'
        df = pd.concat([df_ebrains, df])

        unique_codes = df['diag_org'].unique()

        rowss = []
        rng = np.random.default_rng(42)
        for name, _rows in df.groupby('name'):
            row = _rows.iloc[0]
            ds = row['Dataset']
            if ds == 'EBRAINS':
                count = a.eb_count
            else:
                if ds == 'Local(Train)':
                    count = a.train_count
                elif ds == 'Local(Val)':
                    count = a.val_count
                else:
                    raise RuntimeError('Invalid dataset', ds)
                if row['diag_org'] == 'B':
                    count = 2

            # rows = df.loc[np.random.choice(_rows.index, count)]
            # rows = df.iloc[:count]
            # ii = rng.choice(_rows.index, count)
            # ii = _rows.index[:count]
            # rows = df.loc[ii]
            rows = _rows.head(count)
            if count != len(rows):
                print(name, row['diag_org'], count, len(rows))
            rowss.append(rows)
        df = pd.concat(rowss)
        df = df.reset_index(drop=True)

        df_val = df[df['Dataset'] == 'Local(Val)']
        df_train = df[df['Dataset'] == 'Local(Train)']
        df_eb = df[df['Dataset'] == 'EBRAINS']
        print('total:', len(df))
        print('train:', len(df_train))
        print('val:', len(df_val))
        print('ebrains:', len(df_eb))

        reducer = UMAP(
            n_neighbors=a.n_neighbors,
            min_dist=a.min_dist,
            spread=a.spread,
            n_components=2,
            # n_jobs=1,
            # random_state=42,
        )

        print('Start projection')
        features = np.stack(df['feature'])
        embedding = reducer.fit_transform(features)
        df['UMAP1'] = embedding[:, 0]
        df['UMAP2'] = embedding[:, 1]

        df = df.rename(columns={'diag_org': 'Diagnosis'})
        df = df.drop(columns='feature')
        suffix = f'_{a.n_neighbors}_{a.min_dist}_{a.spread}' if a.with_suffix else ''
        os.makedirs(J(a.model_dir, 'umap'), exist_ok=True)
        df.to_excel(with_wrote(J(a.model_dir, 'umap', f'integrated_embeddings{suffix}.xlsx')))

        if a.show:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111)
            fig.suptitle(a.model_dir)
            g = sns.scatterplot(
                data=df,
                x='UMAP1', y='UMAP2',
                hue='Diagnosis',
                style='Dataset',
                markers={'Local(Train)': 'o', 'Local(Val)': '^', 'EBRAINS': 'X'},
                hue_order=unique_codes,
                palette='tab10',
                s=30,
                alpha=0.8,
                ax=ax,
            )
            if a.hover:
            # if True:
                import resource
                soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                resource.setrlimit(resource.RLIMIT_NOFILE, (100000, hard))
                ii = [Image.open(v).copy() for v in df['path']]
                hover_images_on_scatters([g.collections[0]], [ii], ax=ax)
            plt.show()

    class PredictArgs(CommonArgs):
        model_dir: str = Field(..., s='-d')
        src: str
        dest: str = 'tmp'
        batch_size: int = Field(500, s='-B')

    def run_predict(self, a:PredictArgs):
        checkpoint = Checkpoint.from_file(J(a.model_dir, 'checkpoint_last.pt'))
        config = TrainerConfig.from_file(J(a.model_dir, 'config.json'))

        model = CompareModel(num_classes=config.num_classes(),
                             frozen=config.encoder == 'frozen',
                             base=config.base)
        model.load_state_dict(checkpoint.model_state)
        model = model.to(a.device).eval()

        images, paths = load_images_from_dir_or_file(a.src, with_path=True)
        names = [os.path.basename(p) for p in paths]

        transform = transforms.Compose([
            transforms.CenterCrop(config.crop_size),
            transforms.Resize(config.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

        data = []
        for name, image, path in zip(names, images, paths):
            gg = grid_split(image, size=512,  overwrap=False, flattern=True)
            for g in gg:
                t = transform(g)
                data.append({
                    'name': name,
                    'path': path,
                    'image': g,
                    'tensor': t,
                })

        df = pd.DataFrame(data)
        df[['pred', 'L', 'M', 'G', 'A', 'O', 'B']] = '_'

        num_chunks = math.ceil(len(df) / a.batch_size)
        tq = tqdm(range(num_chunks))

        featuress = []
        for chunk in tq:
            i0 = chunk*a.batch_size
            i1 = (chunk+1)*a.batch_size
            rows = df[i0:i1]
            tt = torch.stack(rows['tensor'].tolist())
            preds = model(tt.to(a.device), activate=True).cpu().detach()
            pred_labels = np.array(list('LMGAOB'))[torch.argmax(preds, dim=1)]
            df.loc[df.index[i0:i1], 'pred'] = pred_labels
            df.loc[df.index[i0:i1], list('LMGAOB')] = preds.numpy()

        df.drop('tensor', axis=1, inplace=True)

        results = []
        for name, rows in df.groupby('name'):
            pred = ''.join(rows['pred'])
            results.append({
                'name': name,
                'path': rows['path'].iloc[0],
                'pred': pred,
                'pred_aggr': max(list(set(pred)), key=pred.count),
                **{
                    k: rows[k].mean() for k in list('LMGAOB')
                }
            })


        df = pd.DataFrame(results)
        print(df)
        df.to_excel(with_wrote('tmp/out.xlsx'))




if __name__ == '__main__':
    cli = CLI()
    cli.run()
