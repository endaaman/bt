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

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 1_000_000_000

IMAGE_CACHE = {}

DIAG_TO_NUM = OrderedDict([(c, i) for i, c in enumerate('LMGAOB')])
NUM_TO_DIAG = list(DIAG_TO_NUM.keys())

# MEAN = [0.8032, 0.5991, 0.8318]
# STD = [0.1203, 0.1435, 0.0829]
# MEAN = np.array([216, 172, 212]) / 255
# STD = np.array([34, 61, 30]) / 255
MEAN = 0.7
STD = 0.1

DEFAULT_SIZE = 512


class Item(NamedTuple):
    path: str
    diag: str
    image: ImageType
    name: str
    grid_index: str
    test: bool


class GridRandomCrop(A.RandomCrop):
    def apply(self, img, h_start=0, w_start=0, **params):
        y = select_side(img.shape[0], self.height)
        x = select_side(img.shape[1], self.width)
        return albuF.crop(img, x_min=x, y_min=y, x_max=x+self.width, y_max=y+self.height)

def aug_train(crop_size, input_size):
    return [
        A.RandomCrop(width=crop_size, height=crop_size),
        A.Resize(width=input_size, height=input_size),
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

def aug_test(crop_size, input_size):
    return [
        A.CenterCrop(width=crop_size, height=crop_size),
        A.Resize(width=input_size, height=input_size),
    ]


class BaseBrainTumorDataset(Dataset):
    def __init__(self,
                 # data spec
                 target='train', source_dir='datasets/LMGAO/images', code='LMGAO',
                 # train-test spec
                 test_ratio=0.25, seed=None,
                 # image spec
                 grid_size=DEFAULT_SIZE, crop_size=DEFAULT_SIZE, input_size=DEFAULT_SIZE,
                 aug_mode='same', normalize=True,
                 skip='',
                 ):
        assert re.match('^[LMGAOB_]{6}$', code)
        self.target = target
        self.code = [*code]
        self.unique_code = [c for c in dict.fromkeys(self.code) if c in 'LMGAOB']
        self.source_dir = source_dir
        self.test_ratio = test_ratio
        self.seed = seed or get_global_seed()
        self.grid_size = grid_size
        self.crop_size = crop_size
        self.input_size = input_size
        self.aug_mode = aug_mode
        self.normalize = normalize
        self.skip = {k:int(v) for k, v in dict(zip(*[iter([*skip])]*2)).items()}

        self.aug_mode = aug_mode
        self.normalize = normalize

        augs = {}
        augs['train'] = aug_train(crop_size, input_size)
        augs['test'] = aug_test(crop_size, input_size)
        augs['all'] = augs['test']

        # select aug
        if aug_mode == 'same':
            aug = augs[target]
        elif aug_mode == 'none':
            aug = []
        else:
            aug = augs[aug_mode]

        if normalize:
            aug += [A.Normalize(mean=MEAN, std=STD)]
        aug += [ToTensorV2()]

        self.albu = A.Compose(aug)
        self.df = self.load_df()


    def load_df(self):
        data = []
        dropped_count = 0
        total = 0
        for original_diag in NUM_TO_DIAG:
            for path in glob(os.path.join(self.source_dir, original_diag, '*.jpg')):
                total += 1
                original_diag_num = DIAG_TO_NUM[original_diag]
                new_diag = self.code[original_diag_num]
                if new_diag == '_':
                    dropped_count += 1
                    continue
                data.append({
                    'path': path,
                    'diag': new_diag,
                    'test': False,
                })
        print(f'total:{total} loaded:{len(data)} dropped:{dropped_count}')

        df_all = pd.DataFrame(data)
        assert len(df_all) > 0, 'NO IMAGES FOUND'
        df_train, df_test = train_test_split(df_all, test_size=self.test_ratio, stratify=df_all['diag'], random_state=self.seed)
        df_test['test'] = True

        if self.target == 'train':
            df = df_train
        elif self.target == 'test':
            df = df_test
        elif self.target == 'all':
            df = pd.concat([df_train, df_test])
        else:
            raise ValueError(f'invalid target: {self.target}')
        return df

    def split_by_grid(self, grid_size):
        items = []
        for _idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            name = os.path.splitext(os.path.basename(row.path))[0]
            org_img = Image.open(row.path)
            if grid_size < 0:
                imgss = [[org_img]]
            else:
                imgss = grid_split(org_img, grid_size, overwrap=False)
            org_img.close()

            offset = self.skip.get(row.diag, 1)
            del org_img
            i = 0
            for h, imgs  in enumerate(imgss):
                for v, img in enumerate(imgs):
                    if i % offset == 0:
                        items.append(Item(
                            path=row.path,
                            diag=row.diag,
                            image=img,
                            name=name,
                            grid_index=f'{h}_{v}',
                            test=row.test,
                        ))
                    else:
                        img.close()
                    i += 1
            del imgss
        print(f'{self.target} images loaded')
        return items

    def as_xy(self, item):
        x = self.albu(image=np.array(item.image))['image']
        y = torch.tensor(self.unique_code.index(item.diag))
        return x, y



class BrainTumorDataset(BaseBrainTumorDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.items = self.split_by_grid(self.grid_size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx % len(self.items)]
        return self.as_xy(item)



class BatchedBrainTumorDataset(BaseBrainTumorDataset):
    def __init__(self, batch_size=16, target_count=8, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.target_count = target_count
        self.items = self.split_by_grid(self.grid_size)

        bg_items = [i for i in self.items if i.diag == 'B']

        self.batched_items = []
        self.batched_labels = []
        for c in self.code:
            target_items = [i for i in self.items if i.diag == c]
            if c == 'B':
                bg_idx = np.random.choice(len(bg_items), self.batch_size)
                bg_ii = [bg_items[i] for i in bg_idx]
                self.batched_items.append(bg_ii)
                self.batched_labels.append('B')

            l = len(target_items)
            idxs = np.array_split(np.random.permutation(np.arrange(l)), l//self.target_count)

            for idx in idxs:
                target_ii = [target_items[i] for i in idx]
                bg_count = batch_size - len(idx)
                bg_idx = np.random.choice(len(bg_items), bg_count)
                bg_ii = [bg_items[i] for i in bg_idx]
                ii = target_ii + bg_ii
                self.batched_items.append(ii)
                self.batched_labels.append(c)

    def __len__(self):
        return len(self.batched_items)

    def __getitem__(self, idx):
        items = self.batched_items[idx]
        label = self.batched_labels[idx]
        xx = []
        for item in items:
            x = self.albu(image=np.array(item.image))['image']
            xx.append(x)
        xx = torch.stack(xx)
        y = torch.tensor(self.unique_code.index(item.diag))
        return xx, y


class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        source_dir: str = 'datasets/LMGAO/images'
        code: str = 'LMGAO'
        target: str = Field('all', cli=('--target', '-t'), choices=['all', 'train', 'test'])
        aug: str = Field('same', cli=('--aug', '-a'), choices=['same', 'train', 'test'])
        crop_size: int = Field(DEFAULT_SIZE, cli=('--crop-size', '-c'))
        input_size: int = Field(DEFAULT_SIZE, cli=('--input-size', '-i'))
        batch_size: int = -1

    def pre_common(self, a):
        if a.batch_size > 0:
            self.ds = BatchedBrainTumorDataset(
                target=a.target,
                source_dir=a.source_dir,
                code=a.code,
                aug_mode=a.aug,
                crop_size=a.crop_size,
                input_size=a.input_size,
                normalize=self.function != 'samples',
                batch_size=a.batch_size,
            )
        else:
            self.ds = BrainTumorDataset(
                target=a.target,
                source_dir=a.source_dir,
                code=a.code,
                aug_mode=a.aug,
                crop_size=a.crop_size,
                input_size=a.input_size,
                normalize=self.function != 'samples',
            )

    class SamplesArgs(CommonArgs):
        dest: str = 'tmp/samples'

    def run_samples(self, a:SamplesArgs):
        t = a.target
        d = os.path.join(a.dest, t)
        os.makedirs(d, exist_ok=True)
        total = len(self.ds)
        for i, (x, y) in tqdm(enumerate(self.ds), total=total):
            img = tensor_to_pil(x)
            item = self.ds.items[i]
            name = os.path.splitext(os.path.basename(item.path))[0]
            img.save(f'{d}/{i}_{NUM_TO_DIAG[int(y)]}_{name}.jpg')
            if i == total-1:
                break

    def run_balance(self, a):
        data = []
        ii = []
        for diag in NUM_TO_DIAG:
            ii.append(diag)
            data.append({
                'pixels': 0,
                'images': 0,
                'mean': 0,
            })
        df = pd.DataFrame(data, index=ii)

        for item in self.ds.items:
            p = item.image.width * item.image.height
            df.loc[item.diag, 'pixels'] += p/1000/1000
            df.loc[item.diag, 'images'] += 1

        for diag in NUM_TO_DIAG:
            m = df.loc[diag, 'pixels'] / df.loc[diag, 'images']
            df.loc[diag, 'mean'] = m
        self.df = df
        chan = '3' if self.a.merge else '5'
        df.to_csv(f'out/balance_{chan}.csv')

    def run_t(self, a):
        for (x, y) in self.ds:
            print(y, x.shape)
            self.x = x
            self.y = y
            self.i = tensor_to_pil(x)
            break

    class GridSplitArgs(CommonArgs):
        dest: str = 'tmp/grid_split'

    def run_grid_split(self, a):
        os.makedirs(self.a.dest, exist_ok=True)
        for item in tqdm(self.ds.items):
            imgss = grid_split(item.image, self.a.crop)
            for h, imgs  in enumerate(imgss):
                for v, img in enumerate(imgs):
                    d = os.path.join(self.a.dest, item.diag, item.name)
                    os.makedirs(d, exist_ok=True)
                    img.save(os.path.join(d, f'{h}_{v}.jpg'))

if __name__ == '__main__':
    # cli = CLI()
    # cli.run()

    # ds = BatchedBrainTumorDataset(batch_size=9, code='LMG__')

    ds = BrainTumorDataset(
        target='all',
        source_dir='datasets/LMGAO/images',
        code='LMGAO',
        aug_mode='same',
        crop_size=512,
        input_size=512,
    )
