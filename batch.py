import os
from glob import glob
import itertools

import pandas as pd
import torch
from PIL import Image, ImageOps, ImageFile
import numpy as np
from tqdm import tqdm
import imagesize

from endaaman import load_images_from_dir_or_file
from endaaman.cli import BaseCLI

from datasets import grid_split
from train import TrainerConfig


class CLI(BaseCLI):
    class CommonArgs(BaseCLI.CommonArgs):
        pass

    class MeanStdArgs(CommonArgs):
        src: str = 'data/images'

    def run_mean_std(self, a):
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

    class MeanStdSimpleArgs(CommonArgs):
        src: str = 'data/images'

    def run_mean_std_simple(self, a):
        pp = sorted(glob(os.path.join(self.a.src, '*/*.jpg')))
        mm = []
        ss = []
        for p in tqdm(pp):
            i = np.array(Image.open(p))[500:1000, 500:1000, :].reshape(-1, 3)
            size = i.shape[0]
            mm.append(np.mean(i, axis=0) / 255)
            ss.append(np.std(i, axis=0) / 255)

        mean = np.mean(mm, axis=0).tolist()
        std = np.mean(ss, axis=0).tolist()
        print('mean', mean)
        print('std', std)


    class GridSplitArgs(CommonArgs):
        src: str = 'data/images'

    def run_grid_split(self, a):
        ii = load_images_from_dir_or_file(self.a.src)[0]
        imgss = grid_split(ii[0], 400)
        for h, imgs  in enumerate(imgss):
            for v, img in enumerate(imgs):
                img.convert('RGB').save(f'tmp/grid/g{h}_{v}.jpg')

    class PrintCheckpointArgs(CommonArgs):
        src: str

    def run_print_checkpoint(self, a):
        self.c = torch.load(a.src)
        # print(self.c)


    class DetailArgs(CommonArgs):
        pass

    def run_detail(self, a):
        diags = ['L', 'M', 'G', 'A', 'O']
        data_images = []
        cases = {}
        for diag in diags:
            paths = sorted(glob(f'datasets/LMGAO/images/{diag}/*.jpg'))
            for path in paths:
                name = os.path.splitext(os.path.basename(path))[0]
                name, order = name.rsplit('_', 1)
                width, height = imagesize.get(path)
                e = {
                    'name': name,
                    'path': path,
                    'order': order,
                    'diag': diag,
                    'width': width,
                    'height': height,
                }
                data_images.append(e)
                if name in cases:
                    cases[name].append(e)
                else:
                    cases[name] = [e]

        data_cases = []
        for case_name, ee in cases.items():
            total_area = 0
            for e in ee:
                total_area += e['width'] * e['height']
            data_cases.append({
                'case': case_name,
                'diag': ee[0]['diag'],
                'count': len(ee),
                'total_area': total_area,
            })
            print(total_area)

        df_cases = pd.DataFrame(data_cases)
        df_images = pd.DataFrame(data_images)
        with pd.ExcelWriter('out/detail.xlsx', engine='xlsxwriter') as writer:
            df_cases.to_excel(writer, sheet_name='cases')
            df_images.to_excel(writer, sheet_name='images')



if __name__ == '__main__':
    cli = CLI()
    cli.run()
