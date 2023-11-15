import os
import re
from glob import glob

import pandas as pd
import torch
from PIL import Image, ImageOps, ImageFile, ImageDraw, ImageFont
import numpy as np
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from pydantic import Field
import seaborn as sns

from endaaman import with_wrote, grid_split
from endaaman.ml import BaseMLCLI

from datasets.utils import show_fold_diag
from utils import calc_white_area




import os
from matplotlib import pyplot as plt
from pydantic import Field

from endaaman.ml import BaseMLCLI
from .fold import FoldDataset


Image.MAX_IMAGE_PIXELS = None

J = os.path.join

class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        pass

    class BuildDatasetArgs(CommonArgs):
        source: str = 'enda2'
        size: int = 512

    def run_build_dataset(self, a):
        dst_dir = f'cache/{a.source}_{a.size}'
        src_dir = f'data/images/{a.source}'
        ee = []
        for diag in 'LMGAOB':
            print(f'loading {diag}')
            paths = sorted(glob(J(src_dir, f'{diag}/*.jpg')))
            diag_dir = J(dst_dir, diag)
            os.makedirs(diag_dir, exist_ok=True)
            data = {}
            for path in tqdm(paths):
                image_name = os.path.basename(path)
                m = re.match(r'^(.*)_(\d+)\.jpg$', image_name)
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
        source: str = 'cache/enda2_512'
        fold: int
        skf: bool = Field(False, cli=('--skf', ))

    def run_split_dataset(self, a):
        df_tiles = pd.read_excel(J(a.source, 'tiles.xlsx'))
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
        with pd.ExcelWriter(J(a.source, f'folds{a.fold}.xlsx')) as w:
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




    class BaseDatasetArgs(BaseMLCLI.CommonArgs):
        source: str = 'cache/enda2_512/'
        code: str = 'LMGGGB'
        fold: int = 0
        total_fold: int = 5
        limit: int = -1
        upsample: bool = Field(False, cli=('--upsample', ))
        show: bool = Field(False, cli=('--show', ))

    class InspectArgs(BaseDatasetArgs):
        pass

    def run_inspect(self, a):
        ds = FoldDataset(
            fold=a.fold,
            total_fold=a.total_fold,
            code=a.code,
            source_dir=a.source,
            limit=a.limit,
            upsample=a.upsample,
            target='all',
        )
        fig = ds.inspect()
        # plt.savefig(J(a.source, 'balance{a.fold}_{a.code}.png'))
        if a.show:
            plt.show()

    class HistArgs(BaseDatasetArgs):
        pass

    def run_hist(self, a):
        ds = FoldDataset(
            fold=a.fold,
            total_fold=a.total_fold,
            code=a.code,
            source_dir=a.source,
            limit=a.limit,
            upsample=a.upsample,
            target='all',
        )
        fig = plt.figure(figsize=(14, 6))
        sns.countplot(
            ds.df,
            x='name',
            hue='diag',
            order=ds.df['name'].value_counts().index
        )
        plt.xticks(rotation=90)
        plt.show()

if __name__ == '__main__':
    cli = CLI()
    cli.run()
