import os
import re
from glob import glob
import hashlib
import shutil
import zipfile

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import Image, ImageOps, ImageFile, ImageDraw, ImageFont
import numpy as np
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from pydantic import Field
import seaborn as sns
import imagesize

from endaaman import with_wrote, grid_split
from endaaman.ml import tensor_to_pil
from endaaman.ml.cli import BaseMLCLI

from utils import calc_white_area
from . import show_fold_diag, FoldDataset, QuadAttentionFoldDataset, PairedFoldDataset
from .ebrains import EBRAINSDataset



Image.MAX_IMAGE_PIXELS = None

J = os.path.join

async def asave(image, path):
    buffer = BytesIO()
    image.save(buffer, format=os.path.splitext(path)[-1][1:])
    async with aiofiles.open(path, "wb") as file:
        await file.write(image)



class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        pass

    class BuildDatasetArgs(CommonArgs):
        source: str = 'enda4'
        dest: str = 'cache'
        size: int = 512
        code: str = 'LMGAOB'

    def run_build_dataset(self, a:BuildDatasetArgs):
        ds_name = f'{a.source}_{a.size}'
        dest_dir = J(a.dest, ds_name)
        src_dir = f'data/images/{a.source}'
        ee = []
        for diag in a.code:
            print(f'loading {diag}')
            paths = sorted(glob(J(src_dir, f'{diag}/*.jpg')))
            diag_dir = J(dest_dir, diag)
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
                        file_name = f'{tile_order:04d}.jpg'
                        path_name = J(case_dir, file_name)
                        if not os.path.exists(path_name):
                            g.save(path_name)
                        white_area = calc_white_area(g)
                        ee.append({
                            'name': name,
                            'diag': diag,
                            'order': image_order,
                            'y': y,
                            'x': x,
                            'filename': file_name,
                            'original': image_name,
                            'width': g.width,
                            'height': g.height,
                            'area': 1-white_area,
                        })
        df_tiles = pd.DataFrame(ee)
        df_tiles.to_csv(with_wrote(J(dest_dir, 'tiles.csv')))


    class ZipArgs(CommonArgs):
        source: str = 'cache/enda4_512'

    def run_zip(self, a):
        name = a.source.split('/')[-1]
        ff = glob(f'{a.source}/*/*/*.jpg', recursive=True)

        os.makedirs(f'/tmp/{name}', exist_ok=True)
        zf = f'/tmp/{name}/tiles.zip'
        print('making', zf)
        with zipfile.ZipFile(zf, 'w', zipfile.ZIP_STORED) as z:
            for f in tqdm(ff):
                name = os.path.relpath(f, a.source)
                z.write(filename=f, arcname=name)

        with open(zf, 'rb') as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()
            print('sha256   : ' + sha256)


    class SplitDatasetArgs(CommonArgs):
        source: str = 'cache/enda4_512'
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
        source: str = 'enda4_512'
        code: str = 'LMGGGB'
        fold: int = 0
        total_fold: int = 5
        limit: int = -1
        upsample: bool = False
        show: bool = False

    class InspectArgs(BaseDatasetArgs):
        pass

    def run_inspect(self, a):
        ds = FoldDataset(
            source=a.source,
            fold=a.fold,
            total_fold=a.total_fold,
            code=a.code,
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
            source=a.source,
            fold=a.fold,
            total_fold=a.total_fold,
            code=a.code,
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
        # plt.xticks(rotation=90)
        plt.xticks([])
        if a.show:
            plt.show()

    class ExampleArgs(CommonArgs):
        source: str = 'enda4_512'
        code: str = 'LMGGGB'
        show: bool = False
        count:int = 30

    def run_example(self, a):
        ds = FoldDataset(
            fold=0,
            total_fold=5,
            code=a.code,
            source=a.source,
            limit=30,
            upsample=False,
            target='train',
            augmentation=True,
            normalization=False,
        )

        d = 'tmp/example/'
        os.makedirs(d, exist_ok=True)

        idxs = np.random.choice(np.arange(len(ds)), a.count)
        print(idxs)

        # local_crop = T.RandomResizedCrop((image_size, image_size), scale = (0.05, local_upper_crop_scale))
        # global_crop = T.RandomResizedCrop((image_size, image_size), scale = (global_lower_crop_scale, 1.))

        for i, idx in enumerate(tqdm(idxs)):
            # x, img = ds[idx]
            row = ds.df.iloc[idx]
            img = ds.load_from_row(row)

            name = f'{i}_{row["diag_org"]}'
            img.save(J(d, f'{name}_a.jpg'))
            aug = tensor_to_pil(ds.aug(image=np.array(img))['image'])
            aug.save(J(d, f'{name}_b.jpg'))

    def run_quad_attention(self, a):
        ds = QuadAttentionFoldDataset(source_dir='data/tiles/enda4_512')
        l = DataLoader(
            dataset=ds,
            batch_size=1,
            num_workers=4,
            shuffle=True
        )

        for x, gt in l:
            print('s', x.shape)
            print('gt', gt)
            break

    class PickWhiteArgs(CommonArgs):
        min: float = 0.6
        max: float = 0.7
        source_dir: str = Field('data/tiles/enda4_512', l='--source', s='-s')

    def run_pick_white(self, a):
        df = pd.read_excel(J(a.source_dir, 'tiles.xlsx'), index_col=0)

        df = df[df['white_area'] > a.min]
        df = df[df['white_area'] < a.max]
        print(df)
        d = f'tmp/pick_{a.min}_{a.max}'
        os.makedirs(d, exist_ok=True)
        for idx, row in df.iterrows():
            image = Image.open(J(a.source_dir, row['diag'], row['name'], row['filename']))
            image.save(f'{d}/{row["diag"]}_{row["name"]}_{row["order"]}_{row["white_area"]:.2f}.png')



    class RotateAndGridArgs(CommonArgs):
        src: str
        dest: str

    def run_rotate_and_grid(self, a):
        diags = list('LMGAO')

        os.makedirs(a.dest, exist_ok=True)

        for diag in diags:
            print(diag)
            base_dir = J(a.src, diag)
            dest_dir = J(a.dest, diag)
            os.makedirs(dest_dir, exist_ok=True)
            if not os.path.isdir(base_dir):
                print(base_dir, 'does not exist. Skipped.')
                continue
            filenames = os.listdir(base_dir)
            filenames_by_case = {}
            for fn in sorted(filenames):
                m = re.match(r'^(N?\d\d-\d\d\d\d?)_.*\.jpg$', fn)
                if not m:
                    raise RuntimeError('Invalid name', fn)
                case = m[1]
                if case in filenames_by_case:
                    filenames_by_case[case].append(fn)
                else:
                    filenames_by_case[case] = [fn]
                size = getattr
            t = tqdm(filenames_by_case.items())

            for case, fns in t:
                new_index = 1
                for fn in fns:

                    img = Image.open(J(base_dir, fn))
                    gg = grid_split(img, size=3000, overwrap=False, flattern=True)
                    if len(gg) == 1:
                        shutil.copyfile(
                            J(base_dir, fn),
                            J(dest_dir, f'{case}_{new_index:02}.jpg')
                        )
                        new_index += 1
                        continue

                    for i, g in enumerate(gg):
                        if g.height > g.width:
                            # g = g.rotate(90, expand=True)
                            g = g.transpose(Image.ROTATE_90)
                            g = ImageOps.exif_transpose(g)
                        g.save(J(dest_dir, f'{case}_{new_index:02}.jpg'), quality=100)
                        new_index += 1
                t.set_description(case)

    def run_paired(self, a):
        ds = PairedFoldDataset(
                source='enda4_512',
                normalization=False,
                target='test',
                size=256,
                minimum_area=0.6)
        D = 'tmp/simsiam_paired'
        os.makedirs(D, exist_ok=True)
        i = 0
        count = 100
        idx = np.random.choice(len(ds), count)
        for i in range(count):
            x01, y = ds.__getitem__(idx[i])
            x0, x1 = x01[0], x01[1]
            tensor_to_pil(x0).save(f'{D}/{i}_0.png')
            tensor_to_pil(x1).save(f'{D}/{i}_1.png')
            i += 1

    def run_example(self, a):
        ds= FoldDataset('enda4_512', normalization=False, size=256, target='test', augmentation=True)
        os.makedirs('tmp/example', exist_ok=True)
        counts = {c:0 for c in range(5)}
        total = 0
        for x, y in ds:
            i = counts[y]
            if i > 5:
                continue
            img = tensor_to_pil(x)
            counts[y] += 1
            img.save(f'tmp/example/{"LMGAOB"[y]}_{i}.png')
            total += 1
            if total > 5 * 6:
                break



    def run_ebrains(self, a):
        ds= EBRAINSDataset(crop_size=512, patch_size=256, normalization=False)
        os.makedirs('tmp/ebrains', exist_ok=True)
        counts = {c:0 for c in range(5)}
        total = 0
        for x, y, index in ds:
            i = counts[y]
            if i > 5:
                continue
            img = tensor_to_pil(x)
            counts[y] += 1
            img.save(f'tmp/ebrains/{"LMGAOB"[y]}_{i}.png')
            total += 1
            if total > 5 * 6:
                break


    # Ported from batch.py below
    def grid_arrange(self, ggg, col_count=-1):
        if col_count > 0:
            # re-arrange into 2d-list
            assert (len(ggg) % col_count) == 0
            row_count = len(ggg) // col_count
            gg = ggg
            ggg = [gg[y*col_count:y*col_count+col_count] for y in range(row_count)]

        row_images = []
        for y, gg in enumerate(ggg):
            row_image_list = []
            for x, g in enumerate(gg):
                row_image_list.append(np.array(g))
            h = cv2.hconcat(row_image_list)
            row_images.append(h)
        merged_image = Image.fromarray(cv2.vconcat(row_images))
        return merged_image

    class GridArrangeArgs(CommonArgs):
        dir: str
        dest: str
        col: int
        scale: int = 1

    def run_grid_arrange(self, a):
        # for p in glob(J(a.dir, '*')):
        #     name = os.path.splitext(os.path.basename(p))[0]
        #     m = re.match(r'^' + a.name + r'_(\d+)_(\d+)$', name)
        #     y = m[1]
        #     x = m[2]
        #     print(a.name, x, y)

        images = []
        for p in tqdm(sorted(glob(J(a.dir, '*')))):
            i = Image.open(p)
            r = i.resize((i.width//a.scale, i.height//a.scale))
            i.close()
            images.append(r)
        m = self.grid_arrange(images, col_count=a.col)
        m.save(a.dest)

    class ConvertToLandscapeArgs(CommonArgs):
        src: str

    def run_convert_to_landscape(self, a):
        pp = sorted(glob(J(a.src, '*/*.jpg'), recursive=True))
        print(len(pp))
        for p in tqdm(pp):
            w, h = imagesize.get(p)
            if (h < w):
                img = Image.open(p)
                img.rotate(90).save(p)
                img.close()

    class GridWsiArgs(CommonArgs):
        src: str
        level: int

    def run_grid_wsi(self, a):
        files = glob(J(a.src, '*.ndpi'))
        name = os.path.split(a.src)[-1]
        for file in files:
            print(file)
            wsi = OpenSlide(file)
            wsi_name = os.path.splitext(os.path.basename(file))[0]
            dest_dir = with_mkdir(f'tmp/B/{wsi_name}')
            W, H = wsi.level_dimensions[a.level]
            ww = np.array(n_split(W, W//3000))
            hh = np.array(n_split(H, H//3000))
            x = 0
            y = 0
            i = 0
            id = 1
            total = len(ww)*len(hh)
            t = tqdm(total=total)
            for _y, h in enumerate(hh):
                x = 0
                for _x, w in enumerate(ww):
                    tile = wsi.read_region((x, y), a.level, (w, h)).convert('RGB')
                    white_area = calc_white_area(tile, min=210, max=255)
                    t.set_description(f'{i}/{total} {_x} {_y} {w} {h} w:{white_area:.3f}')
                    if white_area < 0.5:
                        tile = adjust_gamma(tile, gamma=1/1.8)
                        tile.save(J(dest_dir, f'{wsi_name}_{id:03d}.jpg'))
                        id += 1
                    t.update(1)
                    x += w
                    i += 1
                y += h
            wsi.close()
            t.close()


    class RenumberArgs(CommonArgs):
        src: str

    def run_renumber(self, a):
        paths = glob(J(a.src, '*.jpg'))
        case_name = os.path.split(a.src)[-1]
        cases = {}
        for path in paths:
            m = re.match(r'^(.*)_\d+\.jpg$', os.path.split(path)[1])
            assert m
            name = m[1]
            if name in cases:
                cases[m[1]].append(path)
            else:
                cases[m[1]] = [path]

        print(list(cases.keys()))

        index = 1
        new_cases = {}
        for name, paths in cases.items():
            random.shuffle(paths)
            print(name, len(paths))
            num_groups = len(paths)//30
            counts = n_split(len(paths), num_groups)
            print(counts)
            cursor = 0
            idxs = random.sample(list(range(len(counts))), 3)
            for idx in idxs:
                count = counts[idx]
                new_cases[f'{case_name}-{index:02d}'] = paths[cursor:cursor+count]
                cursor = cursor+count
                index += 1

        for name, paths in new_cases.items():
            index = 1
            for fn in paths:
                d = f'tmp/new/{case_name}/'
                os.makedirs(d, exist_ok=True)
                shutil.copy(J(fn), J(d, f'{name}_{index:02d}.jpg'))
                index += 1


if __name__ == '__main__':
    cli = CLI()
    cli.run()
