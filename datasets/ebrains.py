import os
import re
from glob import glob
from typing import NamedTuple

from tqdm import tqdm
from PIL import Image
from PIL.Image import Image as ImageType
from torchvision import transforms
from torch.utils.data import Dataset

from endaaman.image import grid_split

from . import as_unique_code, MEAN, STD, J


J = os.path.join
EBRAINS_BASE = 'data/EBRAINS'


class Item(NamedTuple):
    path: str
    x: int
    y: int
    X: int
    Y: int
    label: str
    image: ImageType


class EBRAINSDataset(Dataset):
    def __init__(self, crop_size, patch_size, code='LMGAO', mean=MEAN, std=STD):
        self.items = []
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.mean = mean
        self.std = std
        self.code = [*code]

        self.unique_code = as_unique_code(code)

        dirs = os.listdir(EBRAINS_BASE)
        for dir in dirs:
            m = re.search(r'[LMGAO]', dir)
            if m is None:
                raise RuntimeError('Invalid dir detected:', f'{EBRAINS_BASE}/{dir}')
            label = m.group()
            print(f'loading {label}')
            for path in  tqdm(glob(J(EBRAINS_BASE, dir, '*.jpg'))):
                # skip if startswith "_"
                if os.path.basename(path).startswith('_'):
                    continue
                img = Image.open(path)
                ggg = grid_split(img, crop_size, overwrap=False, flattern=False)
                Y = len(ggg)
                for y, gg in enumerate(ggg):
                    X = len(gg)
                    for x, g in enumerate(gg):
                        self.items.append(Item(
                            path=path,
                            label=label,
                            image=g,
                            x=x,
                            X=X,
                            y=y,
                            Y=Y,
                        ))

        self.transform = transforms.Compose([
            transforms.CenterCrop(self.crop_size),
            transforms.Resize(self.patch_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        t = self.transform(item.image)
        label = self.unique_code.index(item.label)
        return t, label, idx
