import os
from glob import glob
import itertools

import torch
from PIL import Image, ImageOps, ImageFile
import numpy as np
from tqdm import tqdm

from endaaman import Commander, get_images_from_dir_or_file
from datasets import grid_split


class CMD(Commander):
    def arg_mean_std(self, parser):
        parser.add_argument('--src', '-s', default='data/images')

    def run_mean_std(self):
        pp = sorted(glob(os.path.join(self.a.src, '*/*.jpg')))
        mm = []
        ss = []
        scale = 0
        for p in tqdm(pp):
            i = np.array(Image.open(p)).reshape(-1, 3)
            size = i.shape[0]
            mm.append(np.mean(i, axis=0) * size / 255)
            ss.append(np.std(i, axis=0) * size / 255)
            scale += size

        mean = (np.sum(mm, axis=0) / scale).tolist()
        std = (np.sum(ss, axis=0) / scale).tolist()
        print('mean', mean)
        print('std', std)

    def arg_grid_split(self, parser):
        parser.add_argument('--src', '-s', default='data/images')

    def run_grid_split(self):
        ii = get_images_from_dir_or_file(self.a.src)[0]
        imgss = grid_split(ii[0], 400)
        for h, imgs  in enumerate(imgss):
            for v, img in enumerate(imgs):
                img.convert('RGB').save(f'tmp/grid/g{h}_{v}.jpg')

if __name__ == '__main__':
    cmd = CMD()
    cmd.run()
