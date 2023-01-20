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
from gradcam import GradCAMpp
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM, EigenCAM, EigenGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from endaaman import get_images_from_dir_or_file, with_wrote
from endaaman.torch import TorchCommander, Predictor, pil_to_tensor, tensor_to_pil
from endaaman.metrics import MultiAccuracy

from models import ModelId, create_model
from datasets import BrainTumorDataset, MEAN, STD, NUM_TO_DIAG, DIAG_TO_NUM
from utils import grid_split, concat_grid_images_float, overlay_heatmap


USE_NEW_CAM = True

CAMs = {
    'gradcampp': GradCAMPlusPlus,
    'eigencam': EigenCAM,
    'eigengradcam': EigenGradCAM,
}

def build_label_text(pred, gt=None):
    text = ''
    pred_num = np.argmax(pred)
    text += f'Pred: {NUM_TO_DIAG[pred_num]}\n'

    if gt is not None:
        text += f'GT: {NUM_TO_DIAG[gt]}\n'

    for i, v in enumerate(pred):
        text += f'{NUM_TO_DIAG[i]}:{v:.2f} '
    return text, ((gt is None) or pred_num == gt)

def draw_pred_gt(image, font, pred, gt=None):
    org_mode = image.mode
    image = image.convert('RGBA')
    text, correct = build_label_text(pred, gt)
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

class CAM(NamedTuple):
    original: ImageType
    heatmap: ImageType
    masked: ImageType
    pred: list


class MyPredictor(Predictor):
    def prepare(self, **kwargs):
        self.font = ImageFont.truetype('/usr/share/fonts/ubuntu/Ubuntu-R.ttf', 48)
        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def create_model(self):
        model_id = ModelId.from_str(self.checkpoint.model_name)
        model = create_model(model_id).to(self.device)
        model.load_state_dict(self.checkpoint.model_state)
        model.eval()
        return model

    def eval(self, inputs):
        return self.model(inputs.to(self.device), activate=True)

    def cam_image_single(self, image, gt=None, cam_type='gradcampp'):
        t = self.transform_image(image).to(self.device)[None, ...]
        with torch.no_grad():
            pred = self.model(t, activate=True).cpu()[0].tolist()
            pred_idx = np.argmax(pred)

        if USE_NEW_CAM:
            CamClass = CAMs[cam_type.lower()]
            gradcam = CamClass(
                model=self.model,
                target_layers=[self.model.get_cam_layer()],
                use_cuda=self.device=='cuda')
            targets = [ClassifierOutputTarget(pred_idx)]
            mask = torch.from_numpy(gradcam(input_tensor=t, targets=targets))
        else:
            gradcam = GradCAMpp(self.model, self.model.get_cam_layer())
            mask, _ = gradcam(t)

        heatmap, masked = overlay_heatmap(mask, pil_to_tensor(image), alpha=0.3, threshold=0.5)
        heatmap, masked = [draw_pred_gt((tensor_to_pil(i)), self.font, pred, gt) for i in (heatmap, masked)]
        return CAM(image, heatmap, masked, pred)

    def cam_image(self, image, gt=None, cam_type='gradcampp', grid_size=-1):
        if grid_size < 0:
            return self.cam_image_single(image, gt)
        images = grid_split(image, size=grid_size, overwrap=False, flattern=True)
        col_count = image.width // grid_size
        cams = [self.cam_image_single(i, gt) for i in images]
        return CAM(
            image,
            concat_grid_images_float([c.heatmap for c in cams], n_cols=col_count),
            concat_grid_images_float([c.masked for c in cams], n_cols=col_count),
            torch.tensor([c.pred for c in cams]).sum(dim=0).div(len(images)).tolist(),
        )


    def predict_image(self, image, grid_size=-1):
        if grid_size < 0:
            return self.predict_images([image], batch_size=1)[0]
        ii = grid_split(image, size=grid_size, flattern=True)
        preds = self.predict_images(ii, batch_size=self.batch_size)
        return torch.stack(preds).sum(dim=0) / len(preds)


class CMD(TorchCommander):
    def arg_common(self, parser):
        parser.add_argument('--checkpoint', '-p', required=True)

    def pre_common(self):
        self.font = ImageFont.truetype('/usr/share/fonts/ubuntu/Ubuntu-R.ttf', 24)
        self.checkpoint = torch.load(self.args.checkpoint, map_location=lambda storage, loc: storage)
        self.predictor = self.create_predictor(
            P=MyPredictor,
            checkpoint=self.checkpoint,
        )
        self.num_classes = self.predictor.model.num_classes

    def arg_dataset(self, parser):
        parser.add_argument('--target', '-t', choices=['test', 'train', 'all'], default='all')
        parser.add_argument('--src-dir', '-s', default='data/images')
        parser.add_argument('--grid-size', '-g', type=int, default=-1)
        parser.add_argument('--name', '-n', default='report')

    def run_dataset(self):
        dataset = BrainTumorDataset(
            target=self.args.target,
            src_dir=self.a.src_dir,
            class_map=='LMGGG' if self.num_classes == 3 else 'LMGAO',
            normalize=False,
            aug_mode='none',
            grid_size=-1,
        )

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

    def arg_cam(self, parser):
        parser.add_argument('--src', '-s', required=True)
        parser.add_argument('--dest', '-d', default='cam')
        parser.add_argument('--grid-size', '-g', type=int, default=-1)
        cam_keys = list(CAMs.keys())
        parser.add_argument('--cam', '-c', type = lambda s:s.lower(), default=cam_keys[0], choices=cam_keys)

    def run_cam(self):
        images, paths = get_images_from_dir_or_file(self.args.src, with_path=True)
        t = tqdm(zip(images, paths), total=len(images))
        for image, path in t:
            cam = self.predictor.cam_image(image, cam_type=self.a.cam, grid_size=self.a.grid_size)

            dest = os.path.join('out', self.checkpoint.trainer_name, self.args.dest)
            os.makedirs(dest, exist_ok=True)
            name = os.path.splitext(os.path.basename(path))[0]
            cam.heatmap.save(os.path.join(dest, f'{name}_heatmap.jpg'))
            cam.masked.save(os.path.join(dest, f'{name}_masked.jpg'))
            t.set_description(f'processed {name}')
            t.refresh()

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
    cmd = CMD()
    cmd.run()
