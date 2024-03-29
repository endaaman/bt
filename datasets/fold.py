import os
import random
import os.path
import re
import shutil
import itertools
import zipfile
from io import BytesIO
from enum import Enum
from glob import glob
from typing import NamedTuple, Callable
from functools import lru_cache
from collections import OrderedDict
from pydantic import Field
from matplotlib import pyplot as plt
import seaborn as sns

import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch.nn import functional as F
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageFile
from PIL.Image import Image as ImageType
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import albumentations.augmentations.crops.functional as albuF
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.augmentations.crops.functional import center_crop

from endaaman.image import load_images_from_dir_or_file, grid_split, select_side
from endaaman.ml import BaseMLCLI, pil_to_tensor, tensor_to_pil, get_global_seed

from .utils import show_fold_diag

ImageFile.LOAD_TRUNCATED_IMAGES = True
# Image.MAX_IMAGE_PIXELS = 1_000_000_000_000
Image.MAX_IMAGE_PIXELS = None

IMAGE_CACHE = {}

DIAG_TO_NUM = OrderedDict([(c, i) for i, c in enumerate('LMGAOB')])
NUM_TO_DIAG = list(DIAG_TO_NUM.keys())

# MEAN = [0.8032, 0.5991, 0.8318]
# STD = [0.1203, 0.1435, 0.0829]
# MEAN = np.array([216, 172, 212]) / 255
# STD = np.array([34, 61, 30]) / 255
MEAN = 0.7
STD = 0.2

DEFAULT_SIZE = 512

J = os.path.join

def get_augs(image_aug, size, normalization, mean, std):
    blur_limit = (3, 5)

    if image_aug:
        aa = [
            # A.RandomCrop(width=size, height=size),
            A.RandomResizedCrop(width=size, height=size, scale=(0.666, 1.5), ratio=(0.95, 1.05), ),
            A.RandomRotate90(p=1),
            A.Flip(p=0.5),

            # Blurs
            A.OneOf([
                A.MotionBlur(blur_limit=blur_limit),
                A.MedianBlur(blur_limit=blur_limit),
                A.GaussianBlur(blur_limit=blur_limit),
                A.Blur(blur_limit=blur_limit),
            ], p=1.0),
            # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=5, p=0.5),

            # Brightness
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Emboss(),
                # A.RandomBrightnessContrast(brightness_limit=0.1),
                A.RandomToneCurve(),
            ], p=1.0),

            # Color
            A.OneOf([
                A.RGBShift(),
                A.HueSaturationValue(sat_shift_limit=20),
            ], p=1.0),

            # Noise
            A.OneOf([
                A.ISONoise(),
                A.GaussNoise(),
                A.ImageCompression(quality_lower=50, quality_upper=100),
            ], p=1.0),

            # Transform
            A.OneOf([
                A.CoarseDropout(max_holes=16, min_holes=1,
                                max_height=32, max_width=32,
                                min_height=8, min_width=8, fill_value=0, p=1.0),
                A.RandomGridShuffle(grid=(2, 2)),
                A.RandomGridShuffle(grid=(3, 3)),
            ], p=1.0),
        ]
    else:
        aa = [
            A.CenterCrop(width=size, height=size),
            # A.Resize(width=size, height=size),
        ]

    if normalization:
        aa += [A.Normalize(mean=mean, std=std)]
    aa += [ToTensorV2()]
    return A.Compose(aa)

@lru_cache
def load_zipfile(path):
    p = os.path.abspath(path)
    with open(p, 'rb') as f:
        return zipfile.ZipFile(BytesIO(f.read()))


class BaseFoldDataset(Dataset):
    def __init__(self,
                 source_dir,
                 total_fold=5,
                 fold=0,
                 target='train',
                 code='LMGAO_',
                 size=512,
                 crop_size=-1,
                 minimum_area=-1,
                 limit=-1,
                 limit_B=300,
                 upsample=False,
                 augmentation=True,
                 normalization=True,
                 mean=MEAN,
                 std=STD,
                 ):
        if len(code) == 5:
            code += '_'
        assert re.match('^[LMGAOB_]{6}$', code)
        self.fold = fold
        self.total_fold = total_fold
        self.source_dir = source_dir
        self.target = target
        self.code = [*code]
        self.size = size
        self.crop_size = crop_size if crop_size > 0 else size
        self.minimum_area = minimum_area
        self.limit = limit
        self.limit_B = limit_B
        self.augmentation = augmentation
        self.normalization = normalization
        self.mean = mean
        self.std = std

        self.unique_code = [c for c in dict.fromkeys(self.code) if c in 'LMGAOB']

        df_tiles = pd.read_excel(J(source_dir, 'tiles.xlsx'), index_col=0)
        df_cases = pd.read_excel(J(source_dir, f'folds{total_fold}.xlsx'), index_col=0)
        df_merge = pd.merge(df_tiles, df_cases.drop(columns='diag'), on='name')
        self.df = df_merge.copy()
        self.df_cases = df_cases.copy()
        self.total_fold = self.df['fold'].max() + 1

        self.zipfile = None
        zf = J(source_dir, 'tiles.zip')
        if os.path.exists(zf):
            print(f'Loading zip archive {zf}')
            self.zipfile = load_zipfile(zf)
            print(f'Loaded {zf}')

        assert self.fold < self.total_fold

        self.aug = get_augs(augmentation, self.crop_size, normalization, mean, std)

        # process like converting LMGAO to LMGGG
        replacer = []
        self.df.loc[:, 'diag_org'] = self.df['diag']
        self.df_cases.loc[:, 'diag_org'] = self.df_cases['diag']
        for old_diag, new_diag in zip('LMGAOB', self.code):
            if old_diag == new_diag:
                continue
            replacer.append([
                new_diag,
                self.df[self.df['diag'] == old_diag].index,
                self.df_cases[self.df_cases['diag'] == old_diag].index,
            ])

        for new_diag, df_idx, df_cases_idx in replacer:
            if new_diag == '_':
                self.df.drop(df_idx, inplace=True)
                self.df_cases.drop(df_cases_idx, inplace=True)
            else:
                self.df.loc[df_idx, 'diag'] = new_diag
                self.df_cases.loc[df_cases_idx, 'diag'] = new_diag

        ##* Filter by area
        if minimum_area > 0:
            # self.df.loc[self.df['white_area'] > minimum_area, 'flag_area'] = False
            # self.df = self.df[self.df['white_area'] < minimum_area].copy()
            self.df = self.df[(self.df['white_area'] < 0.8) | (self.df['diag'] == 'B')].copy()

        ##* Filter by area
        drop_idxs = []
        for i, row in self.df_cases.iterrows():
            is_B = row['diag'] == 'B'
            l = limit_B if is_B else limit
            if l < 0:
                continue
            rows = self.df[self.df['name'] == row.name]
            # rows = rows[rows['flag_area']]
            if len(rows) < l:
                continue
            # random
            # drop_idxs.append(np.random.choice(rows.index, size=len(rows)-l, replace=False))
            # print(np.random.choice(rows.index, size=len(rows)-l, replace=False))

            if is_B:
                # just shuffle
                rows = rows.sample(frac=1)
            else:
                # less white -> drop
                rows = rows.sort_values('white_area', ascending=True)
            drop_idxs.append(rows.index[l:])
        self.df = self.df.drop(np.concatenate(drop_idxs))

        ##* Upsample
        if upsample:
            rows_to_concat = []
            assert limit > 0, 'limit must be positive when upsampling'
            for i, row in self.df_cases.iterrows():
                # do not upsample for B case
                if row['diag'] == 'B':
                    continue
                rows = self.df[self.df['name'] == row.name]
                if len(rows) >= limit:
                    continue
                scale, res = divmod(limit, len(rows))
                rows_to_concat += [rows] * (scale-1) + [rows[:res]]
                # rows_to_concat += [rows] * (scale-1)

            self.df = pd.concat([self.df] + rows_to_concat, ignore_index=True).copy()

        # assign train/val/all
        print(f'loaded {target} for fold {fold}')
        if self.target == 'train':
            self.df = self.df[self.df['fold'] != fold]
            self.df_cases = self.df_cases[self.df_cases['fold'] != fold]
        elif self.target == 'test':
            self.df = self.df[self.df['fold'] == fold]
            self.df_cases = self.df_cases[self.df_cases['fold'] == fold]
        elif self.target == 'all':
            pass
        else:
            raise RuntimeError('Invalid target:', self.target)

        print('Balance: cases')
        show_fold_diag(self.df_cases)
        print('Balance: tiles')
        show_fold_diag(self.df)


    def load_from_row(self, row):
        tile_path = J(row['diag_org'], row['name'], row['filename'])
        if self.zipfile:
            # with zipf.open(image_file_name) as image_file:
            # image_data = BytesIO(self.zipfile.read(tile_path))
            # image = Image.open(image_data)
            with self.zipfile.open(tile_path) as image_file:
                image = Image.open(BytesIO(image_file.read()))
        else:
            image = Image.open(J(self.source_dir, tile_path))
        return image

    def inspect(self):
        folds = self.df_cases['fold'].unique()
        total = len(folds)
        fig, axes = plt.subplots(total, 3, figsize=(12, total*3))
        if len(axes.shape) == 1:
            axes = axes[None, :]

        axes[0, 0].set_title('Cases by diag')
        for i, fold in enumerate(folds):
            f = self.df_cases[self.df_cases['fold'] == fold]
            sns.histplot(f['diag'], ax=axes[i, 0])

        axes[0, 1].set_title('Tiles by case')
        for i, fold in enumerate(folds):
            f = self.df_cases[self.df_cases['fold'] == fold]
            sns.barplot(x=f['diag'], y=f['count'], ax=axes[i, 1])

        axes[0, 2].set_title('Tiles by diag')
        for i, fold in enumerate(folds):
            f = self.df[self.df['fold'] == fold]
            sns.countplot(x=f['diag'], ax=axes[i, 2])
        return fig


class FoldDataset(BaseFoldDataset):
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self.load_from_row(row)
        x = self.aug(image=np.array(image))['image']
        y = torch.tensor(self.unique_code.index(row['diag']))
        return x, y


class DinoFoldDataset(BaseFoldDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.target == 'test':
            aa = [
                A.RandomCrop(width=self.crop_size, height=self.crop_size),
            ]
            if self.normalization:
                aa += [A.Normalize(mean=self.mean, std=self.std)]
            aa += [ToTensorV2()]
            self.aug = A.Compose(aa)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self.load_from_row(row)
        a = np.array(image)
        image.close()
        x1 = self.aug(image=a)['image']
        x2 = self.aug(image=a)['image']
        a = None
        y = torch.tensor(self.unique_code.index(row['diag']))
        return x1, x2, y

class IICFoldDataset(BaseFoldDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.aug = get_augs(True, self.size, self.normalization, self.mean, self.std)

        # self.aug = get_augs(augmentation, self.crop_size, normalization, mean, std)
        self.aug_affine = A.Compose([
            A.RandomResizedCrop(width=self.size, height=self.size, scale=(0.666, 1.5), ratio=(0.95, 1.05), ),
            A.RandomRotate90(p=1),
            A.Flip(p=0.5),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self.load_from_row(row)
        arr = np.array(image)
        image.close()
        image = None
        gt = torch.tensor(self.unique_code.index(row['diag']))

        x = self.aug_affine(image=arr)['image']
        y = self.aug(image=arr)['image']
        arr = None
        return x, y, gt

class QuadAttentionFoldDataset(BaseFoldDataset):
    def __init__(self, *args, **kwargs):
        kwargs['crop_size'] = kwargs.get('size', 512)//2
        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(J(self.source_dir, row['diag_org'], row['name'], row['filename']))
        arr = np.array(image)

        ts = self.crop_size
        w = image.width
        h = image.height
        x = np.random.randint(w - ts*2) if w - ts*2 > 0 else 0
        y = np.random.randint(h - ts*2) if h - ts*2 > 0 else 0

        tiles = [
            arr[:ts+y, :ts+x, :],
            arr[:ts+y, ts+x:, :],
            arr[ts+y:, :ts+x, :],
            arr[ts+y:, ts+x:, :],
        ]

        image.close()
        image = None
        gt = torch.tensor(self.unique_code.index(row['diag']))
        xx = [self.aug(image=t)['image'] for t in tiles]
        xx = torch.stack(xx)
        return xx, gt
