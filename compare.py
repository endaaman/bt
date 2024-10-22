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
from matplotlib import pyplot as plt
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

from models import CompareModel
from datasets import FoldDataset, MEAN, STD
from datasets.ebrains import EBRAINSDataset
from utils import draw_frame


torch.multiprocessing.set_sharing_strategy('file_system')
np.set_printoptions(suppress=True, floatmode='fixed')
J = os.path.join


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
        params = [
            {'params': self.model.base.parameters(), 'lr': self.config.lr * self.config.lr_scaling_factor},
            {'params': self.model.fc.parameters(), 'lr': self.config.lr},
        ]
        return optim.Adam(params)
        # return optim.Adam(self.model.parameters(), lr=self.config.lr)

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

        count: int = -1

    def run_ebrains(self, a:EbrainsArgs):
        map3 = {'L':'L', 'M':'M', 'G':'G', 'A':'G', 'O':'G', 'B':'B'}

        config = TrainerConfig.from_file(J(a.model_dir, 'config.json'))
        print('config:', config)

        unique_code = config.unique_code()

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
        for path, rows in df_patches.groupby('name'):
            assert rows['label'].nunique() == 1
            pred = []
            for c in unique_code:
                pred.append(rows[c].mean())
            pred_idx = np.argmax(pred)
            # if B is major, yield pred from second choice
            # if unique_code[pred_idx] == 'B':
            #     pred_idx = np.argsort(pred)[-2]
            pred_label = unique_code[pred_idx]

            gt_label = rows.iloc[0]['label']
            results_cases.append({
                'path': path,
                'label': gt_label,
                'pred': pred_label,
                'correct': int(gt_label == pred_label),
                'correct3': int(map3[gt_label] == map3[pred_label]),
                **dict(zip(unique_code, pred))
            })
        df_cases = pd.DataFrame(results_cases)

        y_true, y_pred = df_cases['label'], df_cases['pred']
        patch_accuracy = df_patches['correct'].mean()
        report = skmetrics.classification_report(y_true, y_pred, zero_division=0.0, output_dict=True)
        report['patch acc'] = patch_accuracy
        df_report = pd.DataFrame(report).T

        y_true3, y_pred3 = y_true.map(map3), y_pred.map(map3)
        patch_accuracy3 = df_patches['correct3'].mean()
        report3 = skmetrics.classification_report(y_true3, y_pred3, zero_division=0.0, output_dict=True)
        report3['patch acc'] = patch_accuracy3
        df_report3 = pd.DataFrame(report3).T

        with pd.ExcelWriter(with_wrote(J(a.model_dir, 'ebrains.xlsx'))) as w:
            df_patches.to_excel(w, sheet_name='patches', index=False)
            df_cases.to_excel(w, sheet_name='cases', index=False)
            df_report.to_excel(w, sheet_name='report', index=True)
            df_report3.to_excel(w, sheet_name='report3', index=True)

#     class CalcEbrainsArgs(BaseDLArgs):
#         model_dir: str = Field(..., s='-d')

#     def run_calc_ebrains(self, a):
#         config = TrainerConfig.from_file(J(a.model_dir, 'config.json'))
#         df_patches = pd.read_excel(J(a.model_dir, 'ebrains.xlsx'), sheet_name='patches')
#         unique_code = config.unique_code()

#         results_cases = []
#         for path, rows in df_patches.groupby('path'):
#             assert rows['label'].nunique() == 1
#             pred = []
#             for c in unique_code:
#                 pred.append(rows[c].mean())
#             pred_idx = np.argmax(pred)
#             # if B is major, yield pred from second choice
#             # if unique_code[pred_idx] == 'B':
#             #     pred_idx = np.argsort(pred)[-2]
#             pred_label = unique_code[pred_idx]

#             gt_label = rows.iloc[0]['label']
#             results_cases.append({
#                 'path': path,
#                 'label': gt_label,
#                 'pred': pred_label,
#                 'correct': int(gt_label == pred_label),
#                 'correct3': int(map3[gt_label] == map3[pred_label]),
#                 **dict(zip(unique_code, pred))
#             })
#         df_cases = pd.DataFrame(results_cases)

#         y_true, y_pred = df_cases['label'], df_cases['pred']
#         patch_accuracy = df_patches['correct'].mean()
#         report = skmetrics.classification_report(y_true, y_pred, zero_division=0.0, output_dict=True)
#         report['patch acc'] = patch_accuracy
#         df_report = pd.DataFrame(report).T

#         y_true3, y_pred3 = y_true.map(map3), y_pred.map(map3)
#         patch_accuracy3 = df_patches['correct3'].mean()
#         report3 = skmetrics.classification_report(y_true3, y_pred3, zero_division=0.0, output_dict=True)
#         report3['patch acc'] = patch_accuracy3
#         df_report3 = pd.DataFrame(report3).T

#         with pd.ExcelWriter(with_wrote(J(a.model_dir, 'ebrains2.xlsx'))) as w:
#             df_patches.to_excel(w, sheet_name='patches', index=False)
#             df_cases.to_excel(w, sheet_name='cases', index=False)
#             df_report.to_excel(w, sheet_name='report', index=True)
#             df_report3.to_excel(w, sheet_name='report3', index=True)


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
        conds_base = [
            'frozen_gigapath_{}',
            'unfrozen_uni_{}',
            'frozen_uni_{}',
            'unfrozen_ctranspath_{}',
            'frozen_ctranspath_{}',
            'unfrozen_baseline-vit_{}',
            'frozen_baseline-vit_{}',
            'unfrozen_baseline-cnn_{}',
            'frozen_baseline-cnn_{}',
        ]
        labels = [
            'Prov-GigaPath(LP)',
            'UNI(FT)',
            'UNI(LP)',
            'CTransPath(FT)',
            'CTransPath(LP)',
            r'VIT-L$\mathrm{_{IN}}$(FT)',
            r'VIT-L$\mathrm{_{IN}}$(LP)',
            r'ResNet-RS 50$\mathrm{_{IN}}$(FT)',
            r'ResNet-RS 50$\mathrm{_{IN}}$(LP)',
        ]

        metrics_fns = {
            'Accuracy': lambda df: df[df.index == 'accuracy'].iloc[0, 0],
            'Accuracy(Patch)': lambda df: df[df.index == 'patch acc'].iloc[0, 0],
            'F1 score': lambda df: df[df.index == 'macro avg'].iloc[0]['f1-score'],
            'Precision': lambda df: df[df.index == 'macro avg'].iloc[0]['precision'],
            'Recall': lambda df: df[df.index == 'macro avg'].iloc[0]['recall'],
            # 'AUROC ': lambda df: df[df.index == 'auc'].iloc[0, 0],
        }

        target_sheet_name = 'report3' if a.coarse else 'report'

        cache = {}
        def load_df(p):
            if p in cache:
                print
                return cache[p]
            df = pd.read_excel(p, sheet_name=target_sheet_name, index_col=0)
            # cache[p] = df
            return df

        dfs_to_save = []
        limits = [10, 25, 100, 500]

        for limit in limits:
            dfs = []
            for cond, label in zip(conds_base, labels):
                cond = cond.format(limit)
                mm = {}
                for key, fn in metrics_fns.items():
                    m = []
                    for fold in range(5):
                        p = f'out/compare/LMGAOB/fold5_{fold}/{cond}/test/report.xlsx'
                        df = load_df(p)
                        try:
                            value = fn(df)
                        except Exception as e:
                            print('Error in', p)
                            print(df)
                            raise e
                        m.append(value)
                    mm[key] = m

                df = pd.DataFrame(mm)
                # df = pd.concat([pd.DataFrame([{'cond': cond}]*5), df])
                # limit = int(re.match(r'^.*_(\d+)$', cond_org)[1])
                df = pd.DataFrame([{'cond': cond, 'limit': limit, 'label': label}]*5).join(df)
                df = df.reset_index().rename(columns={'index': 'fold'})
                dfs.append(df)
            df = pd.concat(dfs).reset_index(drop=True)
            dfs_to_save.append(df)

        grains = 'coarse' if a.coarse else 'fine'
        with pd.ExcelWriter(with_wrote(f'out/figs/results_{grains}_cv.xlsx')) as w:
            for df, limit in zip(dfs_to_save, limits):
                df.to_excel(w, sheet_name=f'{limit}')

    class SummaryEbrainsArgs(CommonArgs):
        coarse: bool = False

    def run_summary_ebrains(self, a):
        # dump out/figs/results_ebrains.xlsx
        conds = [
            (
                'frozen_gigapath_{}', 'Prov-GigaPath(LP)',
            ),
            (
                'unfrozen_uni_{}', 'UNI(FT)',
            ), (
                'frozen_uni_{}', 'UNI(LP)',
            ), (
                'unfrozen_ctranspath_{}', 'CTransPath(FT)',
            ), (
                'frozen_ctranspath_{}', 'CTransPath(LP)',
            ), (
                'unfrozen_baseline-vit_{}', r'VIT-L$\mathrm{_{IN}}$(FT)',
            ), (
                'frozen_baseline-vit_{}', r'VIT-L$\mathrm{_{IN}}$(LP)',
            ),(
                'unfrozen_baseline-cnn_{}', r'ResNet-RS 50$\mathrm{_{IN}}$(FT)',
            ), (
                'frozen_baseline-cnn_{}', r'ResNet-RS 50$\mathrm{_{IN}}$(LP)',
            ),
        ]

        metrics_fns = {
            'Accuracy': lambda df: df[df.index == 'accuracy'].iloc[0, 0],
            'Accuracy(Patch)': lambda df: df[df.index == 'patch acc'].iloc[0, 0],
            'F1 score': lambda df: df[df.index == 'macro avg'].iloc[0]['f1-score'],
            'Precision': lambda df: df[df.index == 'macro avg'].iloc[0]['precision'],
            'Recall': lambda df: df[df.index == 'macro avg'].iloc[0]['recall'],
            # 'AUROC ': lambda df: df[df.index == 'auc'].iloc[0, 0],
        }

        target_sheet_name = 'report3' if a.coarse else 'report'

        limits = [10, 25, 100, 500]
        results = []
        for limit in limits:
            for cond_base, label in conds:
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
                        r[key] = fn(df)
                    results.append(r)

        df_results = pd.DataFrame(results)

        grains = 'coarse' if a.coarse else 'fine'
        with pd.ExcelWriter(with_wrote(f'out/figs/results_{grains}_ebrains.xlsx')) as w:
            for limit in limits:
                df = df_results[df_results['limit'] == limit]
                df.to_excel(w, sheet_name=f'{limit}')



if __name__ == '__main__':
    cli = CLI()
    cli.run()
