import os
import random
import os.path
import re
import shutil
import itertools
from enum import Enum
from glob import glob
from typing import NamedTuple, Callable
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
import albumentations.augmentations.crops.functional as albuF
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.augmentations.crops.functional import center_crop

from endaaman import grid_split, select_side
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

def get_augs(train, size, image_aug, normalize):
    if train:
        aa = [
            A.RandomCrop(width=size, height=size),
            A.RandomRotate90(p=1),
            A.Flip(p=0.5),
        ]
        if image_aug:
            aa += [
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
    else:
        aa = [
            A.CenterCrop(width=size, height=size),
            # A.Resize(width=size, height=size),
        ]

    if normalize:
        aa += [A.Normalize(mean=MEAN, std=STD)]
    aa += [ToTensorV2()]
    return A.Compose(aa)


class FoldDataset(Dataset):
    def __init__(self,
                 total_fold,
                 fold=0,
                 source_dir='cache/enda2_512',
                 target='train',
                 code='LMGAO',
                 size=512,
                 minimum_area=-1,
                 limit=-1,
                 aug_mode='same',
                 image_aug=False,
                 normalize=True,
                 ):
        assert re.match('^[LMGAO_]{5}$', code)
        self.fold = fold
        self.total_fold = total_fold
        self.source_dir = source_dir
        self.target = target
        self.code = [*code]
        self.size = size
        self.minimum_area = minimum_area
        self.limit = limit
        self.aug_mode = aug_mode
        self.image_aug = image_aug
        self.normalize = normalize

        self.unique_code = [c for c in dict.fromkeys(self.code) if c in 'LMGAO']

        df_tiles = pd.read_excel(J(source_dir, f'tiles.xlsx'), index_col=0)
        df_cases = pd.read_excel(J(source_dir, f'folds{total_fold}.xlsx'), index_col=0)
        df_merge = pd.merge(df_tiles, df_cases.drop(columns='diag'), on='name')
        self.df = df_merge.copy()
        self.df_cases = df_cases.copy()

        self.total_fold = self.df['fold'].max() + 1

        assert self.fold < self.total_fold

        augs = {}
        augs['train'] = self.aug_train = get_augs(True, self.size, image_aug, normalize)
        augs['test'] = self.aug_test = get_augs(False, self.size, False, normalize)
        augs['all'] = augs['test']

        if aug_mode == 'same':
            self.aug = augs[target]
        else:
            self.aug = augs[aug_mode]

        # process like converting LMGAO to LMGGG
        replacer = []
        self.df.loc[:, 'diag_org'] = self.df['diag']
        self.df_cases.loc[:, 'diag_org'] = self.df_cases['diag']
        for old_diag, new_diag in zip('LMGAO', self.code):
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
                self.df_cases.drop(df_idx, inplace=True)
            else:
                self.df.loc[df_idx, 'diag'] = new_diag
                self.df_cases.loc[df_cases_idx, 'diag'] = new_diag


        # filter by area
        self.df['flag_area'] = True
        if minimum_area > 0:
            self.df.loc[self.df['white_area'] > minimum_area, 'flag_area'] = False

        self.df['flag_limit'] = True
        if limit > 0:
            for i, row in self.df_cases.iterrows():
                rows = self.df[self.df['name'] == row.name]
                rows = rows[rows['flag_area']]
                if len(rows) < limit:
                    continue
                drop_idx = np.random.choice(rows.index, size=len(rows)-limit, replace=False)
                # drop_idx = rows.sort_values('white_area')
                self.df.loc[drop_idx, 'flag_limit'] = False

        self.df = self.df[self.df['flag_limit'] & self.df['flag_area']].copy()

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


    def inspect(self, df=None):
        if df is None:
            df = self.df_cases
        folds =df['fold'].unique()
        total = len(folds)
        fig, axes = plt.subplots(total, 2, figsize=(6, total*2))
        if len(axes.shape) == 1:
            axes = axes[None, :]
        for i, fold in enumerate(folds):
            f = df[df['fold'] == fold]
            sns.histplot(f['diag'], ax=axes[i, 0])

        for i, fold in enumerate(folds):
            f = df[df['fold'] == fold]
            sns.barplot(x=f['diag'], y=f['count'], ax=axes[i, 1])
        return fig

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(J(self.source_dir, row['diag_org'], row['name'], row['filename']))

        x = self.aug(image=np.array(image))['image']
        y = torch.tensor(self.unique_code.index(row['diag']))
        return x, y
