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
from matplotlib import pyplot as plt
import seaborn as sns
# import torch_optimizer as optim2
from torchvision.utils import make_grid
from sklearn import metrics as skmetrics
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pydantic import Field
from timm.scheduler.cosine_lr import CosineLRScheduler
import pytorch_grad_cam as CAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from endaaman import load_images_from_dir_or_file, with_wrote, grid_split
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
    model_name:str
    grid_size: int
    crop_size: int
    input_size: int
    source: str
    skip: str
    mil: bool = False
    mil_target_count: int = -1
    mil_activation: str = -1
    mil_param_count: int = -1


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
        if self.config.mil:
            return AttentionModel(
                name=self.config.model_name,
                num_classes=num_classes,
                activation=self.config.mil_activation,
                params_count=self.config.mil_param_count,
            )
        return TimmModel(name=self.config.model_name, num_classes=num_classes)

    def eval(self, inputs, gts):
        if self.config.mil:
            inputs = inputs[0]
            gts = gts[0]
        preds = self.model(inputs.to(self.device), activate=False)
        loss = self.criterion(preds, gts.to(self.device))
        if self.config.mil:
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


    class CommonTrainArgs(CommonArgs):
        lr: float = 0.0002
        batch_size: int = Field(16, cli=('--batch-size', '-B', ))
        num_workers: int = 4
        epoch: int = 10
        model_name: str = Field('tf_efficientnetv2_b0', cli=('--model', '-m'))
        source: str = Field('images', cli=('--source', ))
        suffix: str = ''
        grid_size: int = Field(512, cli=('--grid-size', '-g'))
        crop_size: int = Field(512, cli=('--crop-size', '-c'))
        input_size: int = Field(512, cli=('--input-size', '-i'))
        size: int = Field(-1, cli=('--size', '-s'))
        code: str = 'LMGAO_'
        loss_weights: str = '10'
        experiment_name:str = Field('Default', cli=('--exp', ))
        overwrite: bool = Field(False, cli=('--overwrite', '-o'))
        skip:str = ''

    class TrainArgs(CommonTrainArgs):
        pass

    def run_train(self, a:TrainArgs):
        config = TrainerConfig(
            batch_size=a.batch_size,
            num_workers=a.num_workers,
            lr=a.lr,
            code=a.code,
            model_name=a.model_name,
            loss_weights=a.loss_weights,
            grid_size=a.size if a.size > 0 else a.grid_size,
            crop_size=a.size if a.size > 0 else a.crop_size,
            input_size=a.size if a.size > 0 else a.input_size,
            source=a.source,
            skip=a.skip,
            mil=False,
            mil_count=-1,
        )

        source_dir = J('datasets/LMGAO', a.source)

        dss = [
            BrainTumorDataset(
                target=t,
                source_dir=source_dir,
                code=a.code,
                aug_mode='same',
                crop_size=config.crop_size,
                input_size=config.input_size,
                seed=a.seed,
                skip=a.skip,
            ) for t in ('train', 'test')
        ]

        out_dir = f'out/{a.experiment_name}/{a.config.code}/{config.model_name}_{a.source}'
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


    class MilArgs(CommonTrainArgs):
        target_count: int = Field(8, cli=('--target-count', ))
        activation: str = 'softmax'
        params_count: int = 128

    def run_mil(self, a:MilArgs):
        config = TrainerConfig(
            batch_size=1,
            num_workers=a.num_workers,
            lr=a.lr,
            code=a.code,
            model_name=a.model_name,
            loss_weights=a.loss_weights,
            grid_size=a.size if a.size > 0 else a.grid_size,
            crop_size=a.size if a.size > 0 else a.crop_size,
            input_size=a.size if a.size > 0 else a.input_size,
            source=a.source,
            skip=a.skip,
            mil=True,
            mil_target_count=a.target_count,
            mil_activation=a.activation,
            mil_param_count=a.params_count,
        )

        source_dir = J('datasets/LMGAO', a.source)

        dss = [
            BatchedBrainTumorDataset(
                target=t,
                source_dir=source_dir,
                code=a.code,
                aug_mode='same',
                crop_size=config.crop_size,
                input_size=config.input_size,
                seed=a.seed,
                skip=a.skip,
                batch_size=a.batch_size,
                target_count=a.target_count,
            ) for t in ('train', 'test')
        ]

        out_dir = f'out/{a.experiment_name}/{a.code}_MIL{a.target_count}/{config.model_name}_{a.source}'
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



    class SingleArgs(CommonArgs):
        src: str
        gt: str
        model_dir: str = Field(..., cli=('--model-dir', '-m'))

    def run_single(selfl, a):
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

        gradcam = CAM.GradCAM(
            model=model,
            target_layers=[model.base.layer4[-1].act3],
            use_cuda=False)

        for i, p in zip(ii, pp):
            # t = pil_to_tensor(i)[None, ...]
            batch = transform(i)[None, ...]
            pred = model(batch, activate=True).flatten()
            pred_id = torch.argmax(pred)
            diag = NUM_TO_DIAG[pred_id]
            pred_str = ' '.join([f'{NUM_TO_DIAG[i]}:{p:.2f}' for i, p in enumerate(pred)])

            mask = gradcam(input_tensor=batch, targets=[ClassifierOutputTarget(pred_id)])[0]
            visualization = show_cam_on_image(np.array(i)/255, mask, use_rgb=True)

            fig = plt.figure(figsize=(16, 7))
            fig.add_subplot(1, 2, 1)
            plt.imshow(i)
            fig.add_subplot(1, 2, 2)
            plt.imshow(visualization)

            plt.suptitle(f'GT:{a.gt} pred:{diag} / ' + pred_str)

            name = os.path.splitext(os.path.basename(p))[0]
            print(name, diag, pred_str)

            d = os.path.join(a.model_dir, 'predict')
            os.makedirs(d, exist_ok=True)
            plt.savefig(with_wrote(J(d, f'{name}.jpg')))
            # plt.show()
            plt.close()



    class GridArgs(CommonArgs):
        src: str
        gt: str
        model_dir: str = Field(..., cli=('--model-dir', '-m'))

    def run_grid(selfl, a):
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

        gradcam = CAM.GradCAM(
            model=model,
            target_layers=[model.base.layer4[-1].act3],
            use_cuda=False)

        for image, p in zip(ii, pp):
            # t = pil_to_tensor(i)[None, ...]
            tiles = grid_split(image, 512, overwrap=False, flattern=True)

            tiles = tiles[:12]
            fig = plt.figure(figsize=(16, 7*len(tiles)))
            for i, tile in enumerate(tiles):
                batch = transform(tile)[None, ...]
                pred = model(batch, activate=True).flatten()
                pred_id = torch.argmax(pred)
                diag = NUM_TO_DIAG[pred_id]
                pred_str = ' '.join([f'{NUM_TO_DIAG[i]}:{p:.2f}' for i, p in enumerate(pred)])

                mask = gradcam(input_tensor=batch, targets=[ClassifierOutputTarget(pred_id)])[0]
                vis = show_cam_on_image(np.array(tile)/255, mask, use_rgb=True)
                print(vis.shape)

                fig.add_subplot(len(tiles), 2, i*2+1)
                plt.title(f'GT:{a.gt} pred:{diag}')
                plt.imshow(tile)
                fig.add_subplot(len(tiles), 2, i*2+2)
                plt.imshow(vis)
                plt.title(pred_str)

            name = os.path.splitext(os.path.basename(p))[0]
            print(name, diag, pred_str)

            d = os.path.join(a.model_dir, 'predict')
            os.makedirs(d, exist_ok=True)
            plt.savefig(with_wrote(J(d, f'{name}.jpg')))
            # plt.show()
            plt.close()




if __name__ == '__main__':
    cli = CLI()
    cli.run()
