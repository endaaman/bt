import os
import re
from glob import glob
from typing import NamedTuple
from dataclasses import dataclass, asdict

from tqdm import tqdm
from PIL import Image
from PIL.Image import Image as ImageType
from torchvision import transforms
from torch.utils.data import Dataset

from endaaman.image import grid_split

from . import as_unique_code, MEAN, STD, J


J = os.path.join
EBRAINS_BASE = 'data/EBRAINS'
EBRAINS_CACHE = 'data/EBRAINS/cache'


@dataclass
class Item:
    name: str
    path: str
    x: int
    y: int
    X: int
    Y: int
    label: str
    image: ImageType

    def asdict(self):
        return asdict(self)


class EBRAINSDataset(Dataset):
    def __init__(self, crop_size, patch_size, code='LMGAO', mean=MEAN, std=STD, normalization=True):
        self.items = []
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.mean = mean
        self.std = std
        self.code = [*code]

        self.unique_code = as_unique_code(code)
        self.using_cahce = os.path.exists(EBRAINS_CACHE)

        if self.using_cahce:
            print('Using cache:', EBRAINS_BASE)
        else:
            print('Extracting cache to', EBRAINS_BASE)
            groups = os.listdir(EBRAINS_BASE)
            for group in groups:
                m = re.search(r'[LMGAO]', group)
                if m is None:
                    raise RuntimeError('Invalid dir detected:', f'{EBRAINS_BASE}/{group}')
                label = m.group()
                print(f'loading {label}')
                for path in tqdm(glob(J(EBRAINS_BASE, group, '*.jpg'))):
                    # skip if startswith "_"
                    if os.path.basename(path).startswith('_'):
                        continue
                    img = Image.open(path)
                    ggg = grid_split(img, crop_size, overwrap=False, flattern=False)
                    name = os.path.splitext(os.path.basename(path))[0]

                    os.makedirs(J(EBRAINS_CACHE, label, name), exist_ok=True)
                    Y = len(ggg)
                    for y, gg in enumerate(ggg):
                        X = len(gg)
                        for x, g in enumerate(gg):
                            g.save(J(EBRAINS_CACHE, label, name, f'{Y}-{y}_{X}-{x}.png'))

        has_file = False
        for label in self.code:
            paths = glob(J(EBRAINS_CACHE, label, '*/*.png'))
            for path in paths:
                m = re.match(r'^(\d+)-(\d+)_(\d+)-(\d+)\.png$', os.path.basename(path))
                if m is None or len(m.groups()) != 4:
                    raise RuntimeError('Invalid patch name detected:', path)
                Y, y, X, x = [int(v) for v in m.groups()]
                name = os.path.basename(os.path.split(path)[0])
                self.items.append(Item(
                    name=name,
                    path=path,
                    label=label,
                    image=None,
                    x=x,
                    X=X,
                    y=y,
                    Y=Y,
                ))
                has_file = True

        if not has_file:
            raise RuntimeError('May be cache is not properly initialized.')

        augs = [
            transforms.CenterCrop(self.crop_size),
            transforms.Resize(self.patch_size),
            transforms.ToTensor(),
        ]
        if normalization:
            augs += [transforms.Normalize(mean=self.mean, std=self.std)]
        self.transform = transforms.Compose(augs)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        if item.image is None:
            item.image = Image.open(item.path).convert('RGB')
        t = self.transform(item.image)
        label = self.unique_code.index(item.label)
        return t, label, idx


