import json
import os
import re
from glob import glob
from itertools import islice
from typing import NamedTuple, Callable

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.utils import make_grid, save_image
import pandas as pd
from PIL.Image import Image as ImageType
from PIL import Image, ImageDraw, ImageFont
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns
from pydantic import Field
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM, EigenCAM, EigenGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from endaaman import load_images_from_dir_or_file, with_wrote
from endaaman.ml import BaseMLCLI, pil_to_tensor, tensor_to_pil
from endaaman.ml.metrics import MultiAccuracy

from datasets import BrainTumorDataset, MEAN, STD, NUM_TO_DIAG, DIAG_TO_NUM
from utils import grid_split, concat_grid_images_float, overlay_heatmap
from models import TimmModel, AttentionModel

from main import TrainerConfig


J = os.path.join
USE_NEW_CAM = True

CAMs = {
    'gradcampp': GradCAMPlusPlus,
    'eigencam': EigenCAM,
    'eigengradcam': EigenGradCAM,
}

def build_label_text(pred, gt_id=None):
    text = ''
    pred_num = np.argmax(pred)
    text += f'Pred:{NUM_TO_DIAG[pred_num]}'

    if gt_id is not None:
        text += f' GT:{NUM_TO_DIAG[gt_id]}\n'
    else:
        text += '\n'

    for i, v in enumerate(pred):
        text += f'{NUM_TO_DIAG[i]}:{v:.2f} '
    return text, ((gt_id is None) or pred_num == gt_id)

def draw_pred_gt(image, font, pred, gt_id=None):
    org_mode = image.mode
    image = image.convert('RGBA')
    text, correct = build_label_text(pred, gt_id)
    draw = ImageDraw.Draw(image)
    text_args = dict(
        xy=(0, 0),
        text=text,
        align='left',
        font=font,
    )
    rect = draw.textbbox(**text_args)

    fg = Image.new('RGBA', image.size)
    ImageDraw.Draw(fg).rectangle(rect, fill=(0,0,0,128) if correct else (200,0,0,128))
    image = Image.alpha_composite(image, fg)

    draw = ImageDraw.Draw(image)
    draw.text(**text_args, fill='white')
    return image.convert(org_mode)

class Result(NamedTuple):
    original: ImageType
    heatmap: ImageType
    masked: ImageType
    pred: list


class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        model_dir: str = Field(..., cli=('--model-dir', '-m'))
        cpu: bool = Field(False, cli=('--cpu', ))

    def pre_common(self, a):
        self.font = ImageFont.truetype('/usr/share/fonts/ubuntu/Ubuntu-R.ttf', 24)
        self.device = 'cuda' if torch.cuda.is_available() and not a.cpu else 'cpu'
        self.config = TrainerConfig.from_file(J(a.model_dir, 'config.json'))
        num_classes = len(set([*self.config.code]) - {'_'})
        print(self.config)
        if self.config.mil:
            self.model = AttentionModel(
                name=self.config.model_name,
                num_classes=num_classes,
                activation=self.config.mil_activation,
                params_count=self.config.mil_param_count,
            )
        else:
            self.model = TimmModel(name=self.config.model_name, num_classes=num_classes)
        checkpoint = torch.load(J(a.model_dir, 'checkpoint_best.pt'))
        self.model.load_state_dict(checkpoint.model_state)
        self.model.to(self.device)

        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def cam_image_single(self, image, gt_id=None, cam_type='gradcampp'):
        t = self.transform_image(image).to(self.device)[None, ...]
        with torch.no_grad():
            if self.config.mil:
                pred, aa = self.model(t, activate=True, with_attentions=True)
                pred = pred.cpu().tolist()
            else:
                pred = self.model(t, activate=True)
                pred = pred.cpu()[0].tolist()
            pred_idx = np.argmax(pred)

        CamClass = CAMs[cam_type.lower()]
        gradcam = CamClass(
            model=self.model,
            target_layers=self.model.get_cam_layers(),
            # TODO: fix
            # target_layers=[self.model.base.layer4[-1].act3],
            use_cuda=self.device=='cuda')
        targets = [ClassifierOutputTarget(pred_idx)]
        mask = torch.from_numpy(gradcam(input_tensor=t, targets=targets))

        heatmap, masked = overlay_heatmap(mask, pil_to_tensor(image), alpha=0.5, threshold=0.2)
        # attention = torch.sigmoid(attention_logit)
        heatmap, masked = [draw_pred_gt((tensor_to_pil(i)), self.font, pred, gt_id) for i in (heatmap, masked)]
        return Result(image, heatmap, masked, pred)

    def cam_image(self, image, gt_id=None, cam_type='gradcampp', grid_size=-1):
        image = image.crop((0, 0, min(image.width, 512*6), min(image.height, 512*6)))
        if grid_size < 0:
            return self.cam_image_single(image, gt_id)
        images = grid_split(image, size=grid_size, overwrap=False, flattern=True)
        col_count = image.width // grid_size
        results = [self.cam_image_single(i, gt_id) for i in images]
        pred = torch.tensor([c.pred for c in results]).sum(dim=0).div(len(images)).tolist()
        return Result(
            image,
            concat_grid_images_float([r.heatmap for r in results], n_cols=col_count),
            concat_grid_images_float([r.masked for r in results], n_cols=col_count),
            pred,
        )

    class CamArgs(CommonArgs):
        src: str
        dest: str = 'cam'
        grid_size:int = Field(-1, cli=('--grid-size', '-g'))
        gt: str = Field(...)

    def run_cam(self, a:CamArgs):
        images, paths = load_images_from_dir_or_file(a.src, with_path=True)
        t = tqdm(zip(images, paths), total=len(images))
        for image, path in t:
            result = self.cam_image(image, grid_size=a.grid_size, gt_id=DIAG_TO_NUM[a.gt])
            dest = J(a.model_dir, a.dest)
            os.makedirs(dest, exist_ok=True)
            name = os.path.splitext(os.path.basename(path))[0]
            # cam.heatmap.save(os.path.join(dest, f'{name}.jpg'))
            result.masked.save(os.path.join(dest, f'{name}.jpg'))
            t.set_description(f'processed {name}')
            t.refresh()

    class ImageArgs(CommonArgs):
        src: str
        grid_size:int = Field(512, cli=('--grid-size', '-g'))

    def run_image(self, a:ImageArgs):
        ip = load_images_from_dir_or_file(a.src, with_path=True)
        image, path = ip[0][0], ip[1][0]
        images = grid_split(image, size=a.grid_size, overwrap=False, flattern=True)
        for i in images:
            t = self.transform_image(i)
            t = t[None, ...].to(self.device)
            if self.config.mil:
                p, aa = self.model(t, with_attentions=True, activate=True)
                p = p.cpu().detach()
                attention = aa[0].cpu()
                print(p, attention)
            else:
                p = self.model(t, activate=True)
                p = p[0].cpu().detach()
                diag = self.config.code[torch.argmax(p)]
                print(diag, torch.argmax(p), p)


class CMD:
    def arg_dataset(self, parser):
        parser.add_argument('--target', '-t', choices=['test', 'train', 'all'], default='all')
        parser.add_argument('--src-dir', '-s', default='data/images')
        parser.add_argument('--grid-size', '-g', type=int, default=-1)
        parser.add_argument('--name', '-n', default='report')

    def run_dataset(self):
        # dataset = BrainTumorDataset(
        #     target=self.args.target,
        #     src_dir=self.a.src_dir,
        #     class_map=='LMGGG' if self.num_classes == 3 else 'LMGAO',
        #     normalize=False,
        #     aug_mode='none',
        #     grid_size=-1,
        # )

        oo = []
        gt_nums = []
        pred_nums = []
        for item in tqdm(dataset.items):
            pred = self.predictor.predict_image(item.image, grid_size=self.a.grid_size)
            pred_diag = NUM_TO_DIAG[torch.argmax(pred)]
            pred = pred.tolist()
            oo.append({
                'path': item.path,
                'test': int(item.test),
                'gt': item.diag,
                'pred': pred_diag,
                'correct': int(item.diag == pred_diag),
                **{
                    k: pred[l] for k, l in islice(DIAG_TO_NUM.items(), self.num_classes)
                }
            })
            pred_nums.append(np.argmax(pred))
            gt_nums.append(DIAG_TO_NUM[item.diag])
        df = pd.DataFrame(oo)

        t = self.args.target
        base_dir = f'out/{self.checkpoint.trainer_name}'
        os.makedirs(base_dir, exist_ok=True)
        df.to_excel(with_wrote(f'{base_dir}/{self.a.name}_{t}.xlsx'), index=False)

        cm = metrics.confusion_matrix(gt_nums, pred_nums)
        labels = NUM_TO_DIAG[:self.num_classes]
        ax = sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)
        ax.set_ylabel('Ground truth', fontsize=14)
        ax.set_xlabel('Prediction', fontsize=14)
        plt.title(f'Confusion matrix: {t}')
        plt.savefig(f'{base_dir}/{self.a.name}_cm_{t}.png')

    def arg_predict(self, parser):
        parser.add_argument('--src', '-s', required=True)
        parser.add_argument('--grid-size', '-g', type=int, default=-1)

    def run_predict(self):
        images, paths = get_images_from_dir_or_file(self.args.src, with_path=True)
        preds = [self.predictor.predict_image(i, grid_size=self.a.grid_size) for i in images]
        for (path, pred) in zip(paths, preds):
            s = ' '.join([f'{NUM_TO_DIAG[i]}: {v:.3f}' for i, v in enumerate(pred)])
            print(f'{path}: {s}')

    def arg_cam_dataset(self, parser):
        parser.add_argument('--target', '-t', choices=['test', 'train', 'all'], default='all')
        parser.add_argument('--src-dir', '-s', default='data/images')
        parser.add_argument('--dest', '-d', default='cam')
        parser.add_argument('--grid-size', '-g', type=int, default=-1)
        cam_keys = list(CAMs.keys())
        parser.add_argument('--cam', '-c', type = lambda s:s.lower(), default=cam_keys[0], choices=cam_keys)

    def run_cam_dataset(self):
        dataset = BrainTumorDataset(
            target=self.args.target,
            src_dir=self.a.src_dir,
            grid_size=-1,
            normalize=False,
            aug_mode='none',
            merge_G=self.num_classes == 3,
        )

        for item in tqdm(dataset.items):
            t = 'test' if item.test else 'train'
            dest = os.path.join(
                'out',
                self.checkpoint.trainer_name,
                f'{self.a.dest}_{t}',
                item.diag
            )
            os.makedirs(dest, exist_ok=True)
            cam = self.predictor.cam_image(item.image, gt=DIAG_TO_NUM[item.diag], cam_type=self.a.cam, grid_size=self.a.grid_size)
            cam.heatmap.save(os.path.join(dest, f'{item.name}_heatmap.jpg'))
            cam.masked.save(os.path.join(dest, f'{item.name}_masked.jpg'))

if __name__ == '__main__':
    cli = CLI()
    cli.run()
