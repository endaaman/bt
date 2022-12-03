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
from PIL import Image, ImageOps, ImageFile
from PIL.Image import Image as ImageType
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.augmentations.crops.functional import center_crop

ImageFile.LOAD_TRUNCATED_IMAGES = True


Diag = {
    'L': 0,
    'M': 1,
    'X': 2,
}

# MEAN = [0.8032, 0.5991, 0.8318]
# STD = [0.1203, 0.1435, 0.0829]
# MEAN = np.array([216, 172, 212]) / 255
# STD = np.array([34, 61, 30]) / 255
MEAN = [0.807, 0.611, 0.832]
STD = [0.123, 0.147, 0.087]


class Item(NamedTuple):
    path: str
    diag: str
    image: ImageType


class BTDataset(Dataset):
    def __init__(self, target='train', crop_size=768, size=512, aug_mode='same', normalize=True, test_ratio=0.25, seed=42, scale=1):
        self.target = target
        self.size = size
        self.scale = scale
        self.seed = seed
        self.test_ratio = test_ratio

        self.identity = np.identity(len(Diag))

        augs = {}

        augs['train'] = [
            A.RandomCrop(width=crop_size, height=crop_size),
            A.Resize(width=size, height=size),
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

        augs['test'] = [
            A.RandomCrop(width=size, height=size),
        ]

        augs['all'] = augs['test']

        # select aug
        if aug_mode == 'same':
            aug = augs[target]
        else:
            aug = augs[aug_mode]

        if normalize:
            aug += [A.Normalize(mean=MEAN, std=STD)]
        aug += [ToTensorV2()]

        self.albu = A.Compose(aug)
        self.load_data()

    def load_data(self):
        df_all = pd.read_csv('data/labels.csv')
        df_train, df_test = train_test_split(df_all, test_size=self.test_ratio, stratify=df_all.diag, random_state=self.seed)

        if self.target == 'train':
            df = df_train
        elif self.target == 'test':
            df = df_test
        elif self.target == 'all':
            df = df_all
        else:
            raise ValueError(f'invalid target: {self.target}')

        self.df = df
        self.items = []
        for __idx, row in self.df.iterrows():
            self.items.append(Item(
                path=row.path,
                diag=row.diag,
                image=Image.open(row.path),
            ))

    def __len__(self):
        return len(self.items) * self.scale

    def __getitem__(self, idx):
        item = self.items[idx % len(self.items)]
        x = self.albu(image=np.array(item.image))['image']
        y = torch.tensor(Diag[item.diag])
        return x, y


class CMD(Commander):
    def arg_common(self, parser):
        parser.add_argument('--target', '-t', default='all', choices=['all', 'train', 'test'])
        parser.add_argument('--aug', '-a', default='same', choices=['same', 'train', 'test'])
        parser.add_argument('--size', '-s', type=int, default=512)
        parser.add_argument('--csize', '-c', type=int, default=768)

    def pre_common(self):
        self.ds = BTDataset(
            target=self.args.target,
            aug_mode=self.args.aug,
            size=self.args.size,
            normalize=self.args.function != 'samples',
            # normalize=False,
        )

    def run_mean_std(self):
        pp = glob('data/*/*.jpg')
        mm = []
        ss = []
        scale = 0
        for p in tqdm(pp):
            i = np.array(Image.open(p)).reshape(-1, 3)
            size = i.shape[0]
            print(np.mean(i, axis=0))
            mm.append(np.mean(i, axis=0) * size)
            ss.append(np.std(i, axis=0) * size)
            scale += size
            break

        mean = np.round(np.sum(mm, axis=0) / scale)
        std = np.round(np.sum(ss, axis=0) // scale)
        print('mean', mean)
        print('std', std)

    def run_samples(self):
        t = self.args.target
        d = f'tmp/samples/{t}'
        os.makedirs(d, exist_ok=True)
        total = len(self.ds)
        for i, (x, y) in tqdm(enumerate(self.ds), total=total):
            if i > total:
                break
            img = tensor_to_pil(x)
            img.save(f'{d}/{i}_{int(y)}.jpg')

    def run_t(self):
        for (x, y) in self.ds:
            self.x = x
            self.y = y
            print(y, x.shape)
            self.i = tensor_to_pil(x)
            break

if __name__ == '__main__':
    cmd = CMD(options={'no_pre_common': ['split']})
    cmd.run()
