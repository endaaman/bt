import os
import os.path
import re
import shutil
from enum import Enum
from glob import glob
from typing import NamedTuple, Callable
from collections import OrderedDict
from endaaman import Commander
from endaaman.torch import calc_mean_and_std, pil_to_tensor, tensor_to_pil

import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import torchvision.transforms.functional as F
from PIL import Image, ImageOps
from PIL.Image import Image as img
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.augmentations.crops.functional import center_crop


Diag = {
    'L': 0,
    'M': 1,
    'X': 2,
}


MEAN = [0.8032, 0.5991, 0.8318]
STD = [0.1203, 0.1435, 0.0829]

class Item(NamedTuple):
    path: str
    diag: str
    image: img


class BTDataset(Dataset):
    def __init__(self, target='train', size=256, normalize=True, scale=1):
        self.target = target
        self.size = size
        self.scale = scale
        self.identity = np.identity(len(Diag))

        train_augs = [
            A.RandomCrop(width=size, height=size),
            A.RandomRotate90(p=1),
            A.HorizontalFlip(p=0.5),
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=5, p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
        ]

        test_augs = [
            A.RandomCrop(width=size, height=size),
        ]


        common_augs = [ToTensorV2()]
        if normalize:
            common_augs = [A.Normalize(mean=MEAN, std=STD)] + common_augs

        if self.target == 'test':
            self.albu = A.Compose(test_augs + common_augs)
        else:
            self.albu = A.Compose(train_augs + common_augs)

        self.load_data()

    def load_data(self):
        df = pd.read_csv('data/cache/labels.csv', index_col=0)
        if self.target == 'train':
            df = df[df['test'] == 0]
        elif self.target == 'test':
            df = df[df['test'] == 1]

        self.df = df
        self.items = []
        for idx, row in self.df.iterrows():
            self.items.append(Item(
                path=row.path,
                diag=row.diag,
                image=Image.open(row.path),
            ))

    def __len__(self):
        return len(self.items) * self.scale

    def __getitem__(self, idx):
        item = self.items[idx % len(self.items)]

        image = item.image
        x = self.albu(image=np.array(item.image))['image']
        y = torch.tensor(Diag[item.diag])
        return x, y


class C(Commander):
    def arg_split(self, parser):
        parser.add_argument('--ratio', '-r', type=float, default=0.3)

    def run_split(self):
        dd = ['L', 'M', 'X']
        data = []
        for d in dd:
            for p in glob(f'data/{d}/*.jpg'):
                data.append({
                    'diag': d,
                    'path': p,
                    'test': 0,
                })
        df = pd.DataFrame(data)

        df_train, df_test = train_test_split(df, test_size=self.args.ratio, stratify=df.diag)
        df.at[df_test.index, 'test'] = 1
        os.makedirs('data/cache', exist_ok=True)
        p = 'data/cache/labels.csv'
        df.to_csv(p)
        print(f'wrote {p}')

    def arg_common(self, parser):
        parser.add_argument('--target', '-t', default='all')

    def pre_common(self):
        self.ds = BTDataset(
            target=self.args.target,
            normalize=self.args.function != 'samples',
        )

    def run_mean_std(self):
        mean, std = calc_mean_and_std([item.image for item in self.ds.items], dim=[1,2])
        print('mean', mean)
        print('std', std)

    def run_samples(self):
        t = 'test' if self.args.test else 'train'
        d = f'tmp/samples_{t}'
        os.makedirs(d, exist_ok=True)
        for i, (x, y) in enumerate(self.ds):
            if i > len(self.ds):
                break
            self.x = x
            self.y = y
            img = tensor_to_pil(x)
            img.save(f'{d}/{i}_{int(y)}.png')

    def run_t(self):
        for (x, y) in self.ds:
            self.x = x
            self.y = y
            print(y, x.shape)
            self.i = tensor_to_pil(x)
            break

if __name__ == '__main__':
    c = C(options={'no_pre_common': ['split']})
    c.run()
