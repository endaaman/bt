import os
import re
from glob import glob
from typing import NamedTuple
from dataclasses import dataclass, asdict
import pandas as pd

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

def get_ebrains_df():
    mm = []
    for d in sorted(os.listdir('./data/EBRAINS/')):
        m = re.match(r'^\d.*?([A-Z])', d)
        if not m:
            continue
        for filename in os.listdir(f'./data/EBRAINS/{d}'):
            mm.append({
                'name': os.path.splitext(filename)[0],
                'diag': m[1],
                'subdir': d,
                'path': f'./data/EBRAINS/{d}/{filename}',
            })

    df_map = pd.DataFrame(mm)
    df_map = df_map.set_index('name')
    df_map['subtype'] = df_map['subdir'].replace({
        '1. L': 'L',
        '2. M': 'M',
        '3. G_AA-IDH-wild': 'AA, IDH(-)',
        '3. G_DA-IDH-wild': 'DA, IDH(-)',
        '3. G_GBM-IDH-wild_001-100': 'GBM, IDH(-)',
        '3. G_GBM-IDH-wild_101-200': 'GBM, IDH(-)',
        '4. A_AA-IDH-mut': 'AA, IDH(+)',
        '4. A_DA-IDH-mut': 'DA, IDH(+)',
        '4. A_GBM-IDH-mut': 'GBM, IDH(+)',
        '5. O_AO': 'AO',
        '5. O_O': 'O',
    })
    return df_map


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
            groups = sorted(os.listdir(EBRAINS_BASE))
            for group in groups:
                m = re.search(r'[LMGAO]', group)
                if m is None:
                    print('Skip', f'{EBRAINS_BASE}/{group}')
                    # raise RuntimeError('Invalid dir detected:', f'{EBRAINS_BASE}/{group}')
                    continue
                label = m.group()
                print(f'loading {group} for {label}')
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
                            g.save(J(EBRAINS_CACHE, label, name, f'{Y}-{y}_{X}-{x}.jpg'),
                                   quality=100, optimize=True, progressive=True)

        has_file = False
        for label in self.code:
            paths = glob(J(EBRAINS_CACHE, label, '*/*.jpg'))
            for path in paths:
                m = re.match(r'^(\d+)-(\d+)_(\d+)-(\d+)\.jpg$', os.path.basename(path))
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


