import os
import re
from glob import glob
import itertools

import pandas as pd
import torch
from PIL import Image, ImageOps, ImageFile, ImageDraw, ImageFont
import numpy as np
import cv2
from tqdm import tqdm
import imagesize
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from pydantic import Field

from endaaman import load_images_from_dir_or_file, with_wrote, grid_split
from endaaman.ml import BaseMLCLI

from datasets.utils import show_fold_diag
from utils import calc_white_area


Image.MAX_IMAGE_PIXELS = None

J = os.path.join

class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
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
        source:str = 'enda2'
        size:int = 512

    def run_detail(self, a):
        diags = ['L', 'M', 'G', 'A', 'O', 'B']
        data_images = []
        cases = {}
        for diag in diags:
            paths = sorted(glob(f'datasets/LMGAO/images/{a.source}/{diag}/*.jpg'))
            for path in paths:
                name = os.path.splitext(os.path.basename(path))[0]
                name, order = name.rsplit('_', 1)
                width, height = imagesize.get(path)
                patch_count = (width//a.size) * (height//a.size)
                e = {
                    'name': name,
                    'path': path,
                    'order': order,
                    'diag': diag,
                    'width': width,
                    'height': height,
                    'patch_count': patch_count,
                }
                data_images.append(e)
                if name in cases:
                    cases[name].append(e)
                else:
                    cases[name] = [e]

        data_cases = []
        for case_name, ee in cases.items():
            total_area = 0
            patch_count = 0
            for e in ee:
                total_area += e['width'] * e['height']
                patch_count += e['patch_count']
            data_cases.append({
                'case': case_name,
                'diag': ee[0]['diag'],
                'count': len(ee),
                'total_area': total_area,
                'patch_count': patch_count,
            })

        df_cases = pd.DataFrame(data_cases)
        df_images = pd.DataFrame(data_images)
        with pd.ExcelWriter(with_wrote(f'out/detail_{a.source}.xlsx'), engine='xlsxwriter') as writer:
            df_cases.to_excel(writer, sheet_name='cases')
            df_images.to_excel(writer, sheet_name='images')


    class AverageSizeArgs(CommonArgs):
        source: str = 'images'

    def run_average_size(self, a):
        diags = ['L', 'M', 'G', 'A', 'O', 'B']
        data_images = []
        cases = {}
        for diag in diags:
            paths = sorted(glob(f'datasets/LMGAO/{a.source}/{diag}/*.jpg'))
            Ss = []
            for path in paths:
                width, height = imagesize.get(path)
                Ss.append(width*height)
            mean = np.mean(Ss) if len(Ss) > 0 else 0
            print(f'{diag}, {mean:.0f}')


    def run_ds(self, a):
        diags = ['L', 'M', 'G', 'A', 'O', 'B']
        ee = {}
        for diag in diags:
            paths = sorted(glob(f'datasets/LMGAO/images/{diag}/*.jpg'))
            for path in paths:
                filename = os.path.basename(path)
                m = re.match(r'^(.*)_\d*\.jpg$', filename)
                if not m:
                    raise RuntimeError('Invalid', path)
                id = m[1]

                ee[id] = {
                    'case': m[1],
                    # 'filename': filename,
                    'diag': diag,
                }
        df = pd.DataFrame(data=ee.values())

        skf = StratifiedKFold(n_splits=6)
        df['fold'] = -1
        for i, (train_index, test_index) in enumerate(skf.split(df, df['diag'])):
            df.loc[test_index, 'fold'] = i
            print(df.loc[test_index, 'fold'])

        df.to_excel('d.xlsx')



    class DrawAccsArgs(CommonArgs):
        src: str
        render: bool = Field(False, cli=('--render', ))

    def run_draw_accs(self, a):
        model_dir = os.path.dirname(a.src)
        target_name = os.path.splitext(os.path.basename(a.src))[0]
        dest_dir= J(model_dir, target_name,)

        config = TrainerConfig.from_file(J(model_dir, 'config.json'))
        unique_code = list(config.code)

        df = pd.read_excel(a.src)
        df['base_image'] = df['name'].str.cat(df['order'].astype(str), sep='_')

        colors = {
            'L': 'green',
            'M': 'blue',
            'G': 'red',
            'A': 'yellow',
            'O': 'purple',
        }

        font = ImageFont.truetype('/usr/share/fonts/ubuntu/Ubuntu-R.ttf', size=16)

        data = []

        for image_name, items in tqdm(df.groupby('base_image')):
            diag = items.iloc[0]['diag_org']
            name = items.iloc[0]['name']
            if a.render:
                row_images = []
                for base, cols in items.groupby('y'):
                    ii = []
                    for i, item in cols.iterrows():
                        # print(item['x'], item['y'], item['pred'])
                        i = Image.new('RGBA', (item['width'], item['height']), color=(0,0,0,0,))
                        draw = ImageDraw.Draw(i)
                        draw.rectangle(
                            xy=((0, 0), (item['width'], item['height'])),
                            outline=colors[item['pred']],
                        )
                        draw.rectangle(
                            xy=((0, 0), (200, 16)),
                            fill=colors[item['pred']],
                        )
                        draw.text(
                            xy=(0, 0),
                            text=' '.join([f'{k}:{item[k]:.2f}' for k in ['L', 'M', 'G']]),
                            font=font,
                            fill='white'
                        )
                        ii.append(np.array(i))
                    row_image = cv2.hconcat(ii)
                    row_images.append(row_image)
                overlay = Image.fromarray(cv2.vconcat(row_images))
                original_image = Image.open(f'data/images/enda2/{diag}/{image_name}.jpg').convert('RGBA')
                # original_image.paste(overlay)
                original_image = Image.alpha_composite(original_image, overlay)
                d = J(dest_dir, diag)
                os.makedirs(d, exist_ok=True)
                original_image = original_image.convert('RGB')
                original_image.save(J(d, f'{image_name}.jpg'))
                original_image.close()

            preds = items[['L', 'M', 'G']]
            preds_sum = np.sum(preds, axis=0)
            pred_sum = unique_code[np.argmax(preds_sum)]

            print(pred_sum, preds_sum)
            preds_label = np.argmax(preds, axis=1)

            unique_values, counts = np.unique(preds_label, return_counts=True)
            pred_vote = unique_code[unique_values[np.argmax(counts)]]

            data.append({
                'name': name,
                'image_name': image_name,
                'gt': diag,
                'pred(vote)': pred_vote,
                'pred(sum)': pred_sum,
            })

        data = pd.DataFrame(data)
        data.to_excel(J(dest_dir, 'report.xlsx'))



if __name__ == '__main__':
    cli = CLI()
    cli.run()
