import os
import re
from glob import glob

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
from endaaman.torch import TorchCommander, pil_to_tensor
from endaaman.metrics import MultiAccuracy

from models import create_model, available_models
from datasets import BTDataset, MEAN, STD, NumToDiag, DiagToNum



class CMD(TorchCommander):
    def arg_common(self, parser):
        parser.add_argument('--checkpoint', '-c', required=True)

    def pre_common(self):
        self.checkpoint = torch.load(self.args.checkpoint, map_location=lambda storage, loc: storage)
        # self.checkpoint = torch.load(self.args.checkpoint)
        self.model_name = self.checkpoint.name
        self.model = create_model(self.model_name, 3).to(self.device)
        self.model.load_state_dict(self.checkpoint.model_state)
        self.model.eval()

        self.font = ImageFont.truetype('/usr/share/fonts/ubuntu/Ubuntu-R.ttf', 24)

    def predict_images(self, images):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

        outputs = []
        for image in tqdm(images):
            t = transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                o = self.model(t).detach().cpu()
            outputs += o
        return outputs


    def cam_images(self, images):
        target_layer = self.model.get_cam_layer()
        gradcam = GradCAMpp(self.model, target_layer)

        oo = []
        for image in tqdm(images):
            t = F.to_tensor(image)
            normed = F.normalize(t, mean=MEAN, std=STD).unsqueeze(0).to(self.device)
            mask, result = gradcam(normed)
            heatmap, masked = visualize_cam(mask, t)
            oo.append({
                'heatmap': heatmap,
                'masked': masked,
                'result': result[0].cpu().detach(),
            })
        return oo

    def label_to_text(self, t):
        ss = []
        for i, v in enumerate(t):
            label = NumToDiag[i]
            ss.append(f'{label}:{v:.3f}')
        return ' '.join(ss)

    def process_cam_result(self, image, cam, text_hook=None):
        text = self.label_to_text(cam['result'])
        if text_hook:
            text = text_hook(text)
        grid = make_grid([pil_to_tensor(image), cam['heatmap'], cam['masked']], nrow=3)
        grid_image = F.to_pil_image(grid)
        draw = ImageDraw.Draw(grid_image)
        draw.text((20, 20), text, (0, 0, 0), align='left', font=self.font)
        return grid_image

    def arg_dataset(self, parser):
        parser.add_argument('--target', '-t', choices=['test', 'train', 'all'], default='all')

    def run_dataset(self):
        dataset = BTDataset(
            target=self.args.target,
            normalize=False, aug_mode='none'
        )

        results = self.predict_images([i.image for i in dataset.items])

        oo = []
        for (item, result) in zip(dataset.items, results):
            pred = NumToDiag[torch.argmax(result)]
            result = result.tolist()
            oo.append({
                'path': item.path,
                'test': int(item.test),
                'gt': item.diag,
                'pred': pred,
                'correct': int(item.diag == pred),
                'L': result[DiagToNum['L']],
                'M': result[DiagToNum['M']],
                'G': result[DiagToNum['G']],
            })
        df = pd.DataFrame(oo)
        df.to_excel(f'out/{self.model_name}/report_{self.args.target}.xlsx', index=False)

    def load_images_from_dir_or_file(self, src):
        paths = []
        if os.path.isdir(src):
            paths = os.path.join(src, '*.jpg') + os.path.join(src, '*.png')
            images = [Image.open(p) for p in paths]
        elif os.path.isfile(src):
            paths = [src]
            images = [Image.open(src)]

        if len(images) == 0:
            raise RuntimeError(f'Invalid src: {src}')

        return images, paths

    def arg_predict(self, parser):
        parser.add_argument('--src', '-s', required=True)

    def run_predict(self):
        images, paths = self.load_images_from_dir_or_file(self.args.src)

        results = self.predict_images(images)

        for (path, result) in zip(paths, results):
            s = ' '.join([f'{NumToDiag[i]}: {v:.3f}' for i, v in enumerate(result)])
            print(f'{path}: {s}')

    def arg_cam(self, parser):
        parser.add_argument('--src', '-s', required=True)
        parser.add_argument('--dest', '-d', default='cam')

    def run_cam(self):
        images, paths = self.load_images_from_dir_or_file(self.args.src)
        cams = self.cam_images(images)
        dest = os.path.join('out', self.model_name, self.args.dest)
        os.makedirs(dest, exist_ok=True)

        for (path, image, cam) in zip(paths, images, cams):
            name = os.path.splitext(os.path.basename(path))[0]
            grid_image = self.process_cam_result(image, cam)
            grid_image.save(os.path.join(dest, f'{name}.jpg'))
        print('done')

    def arg_cam_dataset(self, parser):
        parser.add_argument('--target', '-t', choices=['test', 'train', 'all'], default='all')
        parser.add_argument('--dest', '-d', default='cam')

    def run_cam_dataset(self):
        dataset = BTDataset(
            target=self.args.target,
            normalize=False, aug_mode='none'
        )

        cams = self.cam_images([i.image for i in dataset.items])
        dest = os.path.join('out', self.model_name, self.args.dest)
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
