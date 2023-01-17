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
from PIL import Image, ImageDraw, ImageFont
from sklearn import metrics
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

from endaaman import get_images_from_dir_or_file
from endaaman.torch import TorchCommander, pil_to_tensor, Predictor
from endaaman.metrics import MultiAccuracy

from models import ModelId, create_model
from datasets import LMGDataset, grid_split, MEAN, STD, NUM_TO_DIAG, DIAG_TO_NUM, MAP5TO3


class CAM(NamedTuple):
    heatmap: torch.Tensor
    masked: torch.Tensor
    result: list

class MyPredictor(Predictor):
    def prepare(self, **kwargs):
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

    def _cam_image(self, image):
        target_layer = self.model.get_cam_layer()
        gradcam = GradCAMpp(self.model, target_layer)
        t = self.transform_image(image).to(self.device)[None, ...]
        mask, result = gradcam(t)
        heatmap, masked = visualize_cam(mask, t)
        return CAM(heatmap, masked, result)

    def cam_image(self, image, grid_size=-1):
        if grid_size < 0:
            return self._cam_image(image)
        images = grid_split(image, size=grid_size, overwrap=False, flattern=True)
        col_count = image.width // grid_size
        cams = [self._cam_image(i) for i in images]
        return CAM(
            make_grid([c.heatmap for c in cams], nrow=col_count),
            make_grid([c.masked for c in cams], nrow=col_count),
            torch.tensor([c.result for c in cams]).sum(dim=0).tolist(),
        )


    def predict_image(self, image, grid_size=-1):
        if grid_size < 0:
            return self.predict_images([image], batch_size=1)[0]
        ii = grid_split(image, size=grid_size, flattern=True)
        results = self.predict_images(ii, batch_size=self.batch_size)
        return torch.stack(results).sum(dim=0) / len(results)


class CMD(TorchCommander):
    def arg_common(self, parser):
        parser.add_argument('--checkpoint', '-c', required=True)

    def pre_common(self):
        self.font = ImageFont.truetype('/usr/share/fonts/ubuntu/Ubuntu-R.ttf', 24)
        self.checkpoint = torch.load(self.args.checkpoint, map_location=lambda storage, loc: storage)
        self.predictor = self.create_predictor(
            P=MyPredictor,
            checkpoint=self.checkpoint,
        )
        self.num_classes = self.predictor.model.num_classes

    def label_to_text(self, t):
        ss = []
        for i, v in enumerate(t):
            label = NUM_TO_DIAG[i]
            ss.append(f'{label}:{v:.3f}')
        return ' '.join(ss)

    def process_cam_result(self, image, cam, text_hook=None):
        text = self.label_to_text(cam['result'])
        if text_hook:
            text = text_hook(text)
        # stack vertical
        grid = make_grid([pil_to_tensor(image), cam['heatmap'], cam['masked']], nrow=1)
        grid_image = F.to_pil_image(grid)
        draw = ImageDraw.Draw(grid_image)
        draw.text((20, 20), text, (0, 0, 0), align='left', font=self.font)
        return grid_image

    def arg_dataset(self, parser):
        parser.add_argument('--target', '-t', choices=['test', 'train', 'all'], default='all')
        parser.add_argument('--grid-size', '-g', type=int, default=-1)
        parser.add_argument('--name', '-n', default='report')

    def run_dataset(self):
        dataset = LMGDataset(
            target=self.args.target,
            normalize=False,
            aug_mode='none',
            grid_size=-1,
            merge_G=self.num_classes == 3,
        )

        oo = []
        for item in dataset.items:
            result = self.predictor.predict_image(item.image, grid_size=self.a.grid_size)
            pred = NUM_TO_DIAG[torch.argmax(result)]
            result = result.tolist()
            oo.append({
                'path': item.path,
                'test': int(item.test),
                'gt': item.diag,
                'pred': pred,
                'correct': int(item.diag == pred),
                **{
                    k: result[l] for k, l in islice(DIAG_TO_NUM.items(), self.num_classes)
                }
            })
        df = pd.DataFrame(oo)
        p = f'out/{self.checkpoint.trainer_name}/{self.a.name}_{self.args.target}.xlsx'
        df.to_excel(p, index=False)
        print(f'wrote {p}')

    def arg_predict(self, parser):
        parser.add_argument('--src', '-s', required=True)
        parser.add_argument('--grid-size', '-g', type=int, default=-1)

    def run_predict(self):
        images, paths = get_images_from_dir_or_file(self.args.src, with_path=True)
        results = [self.predictor.predict_image(i, grid_size=self.a.grid_size) for i in images]
        for (path, result) in zip(paths, results):
            s = ' '.join([f'{NUM_TO_DIAG[i]}: {v:.3f}' for i, v in enumerate(result)])
            print(f'{path}: {s}')

    def arg_cam(self, parser):
        parser.add_argument('--src', '-s', required=True)
        parser.add_argument('--dest', '-d', default='cam')
        parser.add_argument('--grid-size', '-g', type=int, default=-1)

    def run_cam(self):
        images, paths = get_images_from_dir_or_file(self.args.src, with_path=True)
        cam = self.predictor.cam_image(images[0])
        dest = os.path.join('out', self.checkpoint.trainer_name, self.args.dest)
        os.makedirs(dest, exist_ok=True)

        name = os.path.splitext(os.path.basename(path))[0]
        grid_image = self.process_cam_result(image, cam)
        grid_image.save(os.path.join(dest, f'{name}.jpg'))
        print('done')

    def arg_cam_dataset(self, parser):
        parser.add_argument('--target', '-t', choices=['test', 'train', 'all'], default='all')
        parser.add_argument('--dest', '-d', default='cam')

    def run_cam_dataset(self):
        dataset = LMGDataset(
            target=self.args.target,
            normalize=False,
            aug_mode='none'
        )

        cams = self.cam_images([i.image for i in dataset.items])
        dest = os.path.join('out', self.checkpoint.full_name(), self.args.dest)
        os.makedirs(dest, exist_ok=True)

        for (item, cam) in tqdm(zip(dataset.items, cams)):
            name = os.path.splitext(os.path.basename(item.path))[0]
            t = 'test' if item.test else 'train'
            grid_image = self.process_cam_result(
                item.image, cam,
                text_hook=lambda base: f'gt:{item.diag} ({t}) {base}')
            os.makedirs(d:=os.path.join(dest, item.diag), exist_ok=True)
            grid_image.save(os.path.join(d, f'{name}.jpg'))

if __name__ == '__main__':
    cmd = CMD()
    cmd.run()
