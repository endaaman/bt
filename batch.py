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

    class BuildDatasetArgs(CommonArgs):
        source: str = 'enda2'
        size: int = 512
        gen: bool = Field(False, cli=('--gen', ))

    def run_build_dataset(self, a):
        dst_dir = f'cache/{a.source}_{a.size}'
        src_dir = f'data/images/{a.source}'
        ee = []
        for diag in 'LMGAO':
            print(f'loading {diag}')
            paths = sorted(glob(J(src_dir, f'{diag}/*.jpg')))
            diag_dir = J(dst_dir, diag)
            os.makedirs(diag_dir, exist_ok=True)
            data = {}
            for path in tqdm(paths):
                image_name = os.path.basename(path)
                m = re.match(r'^(.*)_(\d)\.jpg$', image_name)
                if not m:
                    raise RuntimeError('Invalid name:', path)
                name, image_order = m[1], m[2]
                i = Image.open(path)
                ggg = grid_split(i, a.size, overwrap=False)
                for y, gg in enumerate(ggg):
                    for x, g in enumerate(gg):
                        if name in data:
                            data[name] += 1
                        else:
                            data[name] = 0
                        tile_order = data[name]
                        case_dir = J(diag_dir, name)
                        os.makedirs(case_dir, exist_ok=True)
                        filename = f'{tile_order:04d}.jpg'
                        g.save(J(case_dir, filename))
                        area = calc_white_area(g)
                        ee.append({
                            'name': name,
                            'diag': diag,
                            'order': image_order,
                            'y': y,
                            'x': x,
                            'filename': filename,
                            'original': image_name,
                            'width': g.width,
                            'height': g.height,
                            'white_area': area,
                        })
        df_tiles = pd.DataFrame(ee)
        df_tiles.to_excel(with_wrote(J(dst_dir, 'tiles.xlsx')))


    class SplitDatasetArgs(CommonArgs):
        src: str = 'cache/enda2_512'
        fold: int = Field(5, cli=('--fold', ))
        skf: bool = Field(False, cli=('--skf', ))

    def run_split_dataset(self, a):
        df_tiles = pd.read_excel(J(a.src, 'tiles.xlsx'))
        cases = []
        for name, d in df_tiles.groupby('name'):
            diag = d['diag'].iloc[0]
            if not np.all(d['diag'] == diag):
                raise RuntimeError('Invalid')
            cases.append({
                'name': name,
                'diag': diag,
                'count': len(d),
            })

        df_cases = pd.DataFrame(cases).sort_values(by=['diag', 'count'])

        if a.skf:
            df_cases['fold'] = -1
            skf = StratifiedKFold(n_splits=a.fold, shuffle=False)
            for i, (train_index, test_index) in enumerate(skf.split(df_cases, df_cases['diag'])):
                df_cases.loc[test_index, 'fold'] = i
        else:
            cycle = np.arange(a.fold)
            # cycle = np.concatenate([cycle, cycle[::-1]])
            l = len(df_cases)
            ids = np.tile(cycle, (l // len(cycle)))
            ids = np.concatenate([ids, cycle[:l % len(cycle)]])
            df_cases['fold'] = ids

        # pylint: disable=abstract-class-instantiated
        with pd.ExcelWriter(J(a.src, f'folds{a.fold}.xlsx')) as w:
            df_cases.to_excel(w, sheet_name='cases', index=False)
            # df_merge.to_excel(w, sheet_name='tiles', index=False)

        df_merge = pd.merge(df_tiles, df_cases.drop(columns='diag'), on='name')
        show_fold_diag(df_cases)
        show_fold_diag(df_merge)


    class DetectWhiteAreaArgs(CommonArgs):
        case: str = '19-0222'

    def run_detect_white_area(self, a):
        pp = glob(f'cache/enda2_512/L/{a.case}/*.jpg')
        dst_dir = f'tmp/show/L/{a.case}'
        os.makedirs(dst_dir, exist_ok=True)
        data = []
        for i, p in enumerate(tqdm(pp)):
            image = np.array(Image.open(p))

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresholded = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 各白色領域の面積を計算
            areas = [cv2.contourArea(cnt) for cnt in contours]

            # 面積の閾値を設定（広いと狭いを区別する閾値）
            large_area_threshold = 1000  # 例: 広いと判断する面積の閾値
            small_area_threshold = 500   # 例: 狭いと判断する面積の閾値

            # 広い領域と狭い領域を区別
            large_areas = [cnt for cnt, area in zip(contours, areas) if area >= large_area_threshold]
            small_areas = [cnt for cnt, area in zip(contours, areas) if area < small_area_threshold]

            ratio = np.max(areas)/gray.shape[0]/gray.shape[1]

            # 広い領域と狭い領域の数を出力
            # print(f"広い領域の数: {len(large_areas)}")
            # print(f"狭い領域の数: {len(small_areas)}")
            # print(f"最大占有率:   {ratio:.3f}")

            # 画像に白色領域を描画して表示（任意）
            cv2.drawContours(image, large_areas, -1, (0, 0, 255), 2)  # 広い領域を赤で描画
            cv2.drawContours(image, small_areas, -1, (0, 255, 0), 2)  # 狭い領域を緑で描画

            # fig = plt.figure()
            # fig.add_subplot(1, 2, 1)
            # plt.imshow(thresholded)

            # fig.add_subplot(1, 2, 2)
            # plt.imshow(image)
            # plt.show()
            cv2.imwrite(J(dst_dir, f'{i:04d}_{ratio*100:.0f}.jpg'), image)
            data.append({
                'number': i,
                'white_area': ratio,
            })
        df = pd.DataFrame(data)
        df.to_excel(J(dst_dir, '_.xlsx'))

    class DrawAccsArgs(CommonArgs):
        src: str
        render: bool = Field(False, cli=('--render', ))

    def run_draw_accs(self, a):
        model_dir = os.path.dirname(a.src)

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
                d = J(model_dir, 'result', diag)
                os.makedirs(d, exist_ok=True)
                original_image = original_image.convert('RGB')
                original_image.save(J(d, f'{image_name}.jpg'))
                original_image.close()

            preds = items[['L', 'M', 'G']]
            preds_sum = np.sum(preds, axis=0)
            pred_sum = list('LMGAO')[np.argmax(preds_sum)]

            print(pred_sum, preds_sum)
            preds_label = np.argmax(preds, axis=1)

            unique_values, counts = np.unique(preds_label, return_counts=True)
            pred_vote = list('LMGAO')[unique_values[np.argmax(counts)]]

            data.append({
                'name': name,
                'image_name': image_name,
                'gt': diag,
                'pred(vote)': pred_vote,
                'pred(sum)': pred_sum,
            })

        data = pd.DataFrame(data)
        data.to_excel(J(model_dir, 'report.xlsx'))



if __name__ == '__main__':
    cli = CLI()
    cli.run()
