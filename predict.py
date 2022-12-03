import os
import re
from glob import glob

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torchvision import transforms
import pandas as pd
from PIL import Image
from sklearn import metrics

from endaaman.torch import Trainer, TorchCommander
from endaaman.metrics import MultiAccuracy

from models import create_model, available_models
from datasets import BTDataset, MEAN, STD, NumToDiag, DiagToNum



class CMD(TorchCommander):
    def arg_common(self, parser):
        parser.add_argument('--weights', '-w', required=True)

    def pre_common(self):
        self.weights = torch.load(self.args.weights, map_location=lambda storage, loc: storage)
        self.model_name = self.weights['name']
        self.model = create_model(self.model_name, 3).to(self.device)
        self.model.load_state_dict(self.weights['state_dict'])

        self.model.to(self.device).eval()

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
                'X': result[DiagToNum['X']],
            })
        df = pd.DataFrame(oo)
        df.to_excel(f'out/{self.model_name}/report_{self.args.target}.xlsx', index=False)

    def arg_predict(self, parser):
        parser.add_argument('--src', '-s', required=True)
        parser.add_argument('--dest', '-d', default='predict')

    def run_predict(self):
        images = []
        src = self.args.src
        paths = []
        if os.path.isdir(src):
            paths = os.path.join(src, '*.jpg') + os.path.join(src, '*.png')
            images = [Image.open(p) for p in paths]
        elif os.path.isfile(src):
            paths = [src]
            images = [Image.open(src)]

        if len(images) == 0:
            raise RuntimeError(f'Invalid src: {src}')

        results = self.predict_images(images)

        for (path, result) in zip(paths, results):
            s = ' '.join([f'{NumToDiag[i]}: {v:.3f}' for i, v in enumerate(result)])
            print(f'{path}: {s}')


if __name__ == '__main__':
    cmd = CMD()
    cmd.run()
