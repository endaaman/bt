import os
import re
from glob import glob
import itertools
import shutil
import hashlib
import random
from itertools import compress

import joblib
import pandas as pd
import torch
from PIL import Image, ImageOps, ImageFile, ImageDraw, ImageFont
import numpy as np
import cv2
from tqdm import tqdm
import imagesize
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.legend_handler import HandlerTuple
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics as skmetrics
from pydantic import Field
from torchvision import transforms
from torchvision.transforms.functional import adjust_gamma
import albumentations as A
from openslide import OpenSlide
import seaborn as sns
import leidenalg as la
import igraph as ig

from endaaman import load_images_from_dir_or_file, with_wrote, grid_split, with_mkdir, n_split
from endaaman.ml.cli import BaseMLCLI
from endaaman.ml.utils import hover_images_on_scatters
import imagesize

from datasets import show_fold_diag
from utils import calc_white_area, draw_frame
from classification import TrainerConfig


Image.MAX_IMAGE_PIXELS = None

J = os.path.join

class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        pass

    def run_balance(self, a):
        df_org = pd.read_excel('data/tiles/enda4_512/folds5.xlsx')
        col_count = 3
        fig, axes = plt.subplots(2, col_count, figsize=(8, 5))

        colors = [
            'tab:green',
            'tab:blue',
            'tab:orange',
        ]

        code = {
            'G': 'G',
            'A': 'G',
            'O': 'G',
            'L': 'L',
            'M': 'M',
            'B': None,
        }
        print(axes)
        for fold in range(5):
            ax = axes[fold//col_count, fold%col_count]
            ax.set_title(f'Fold {fold+1}')
            df = df_org[df_org['fold'] == fold]
            counts = df.value_counts('diag')
            counts2 = {}
            for c, alt in code.items():
                if not alt:
                    continue
                counts2[alt] = counts2.get(alt, 0) + counts[c]
            ax.bar(counts2.keys(), counts2.values(), color=colors)
            # sns.histplot(f['diag'], ax=)
        plt.subplots_adjust()
        plt.tight_layout()
        plt.show()

    def run_population(self, a):
        df = pd.read_excel('data/tiles/enda4_512/folds5.xlsx')
        count = df['diag'].value_counts()
        code = list('GAOLM')
        count = [count[count.index == c].iloc[0] for c in code]
        total = sum(count)
        colors = [
            'tab:green',
            'tab:red',
            'tab:purple',
            'tab:blue',
            'tab:orange',
            # 'tab:brown',
        ]
        labels = [
            'GBM',
            'A',
            'O',
            'L',
            'M',
        ]
        wedges, texts, autotexts = plt.pie(
                count, labels=labels, autopct=lambda p: f'{round(p*total/100)} ({round(p)}%)',
                startangle=90, counterclock=False, colors=colors)

        for text in texts:
            text.set_fontsize(20)
        for text in autotexts:
            text.set_fontsize(12)
            text.set_color('white')
        plt.show()

    class MeanStdArgs(CommonArgs):
        src: str = 'data/images'

    def run_mean_std(self, a):
        pp = glob(os.path.join(self.a.src, '*/*.jpg'))
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


    class HashImagesArgs(CommonArgs):
        dir: str = 'cache/enda2_512'

    def run_hash_images(self, a):
        total_hash = ''
        BUF_SIZE = 65536
        for p in tqdm(sorted(glob(J(a.dir, '**/*.jpg'), recursive=True))):
            hasher = hashlib.new('sha1')
            with open(p, 'rb') as f:
                while True:
                    data = f.read(BUF_SIZE)
                    if not data:
                        break
                    hasher.update(data)
            total_hash += str(hasher.hexdigest())

        hasher = hashlib.new('sha1')
        hasher.update(total_hash.encode('utf-8'))
        print(hasher.hexdigest())


    class DrawResultArgs(CommonArgs):
        model_dir: str = Field(..., s='-d')
        target: str = Field('test', s='-t')
        count: int = 1

    def run_draw_result(self, a):
        dest_dir= J(a.model_dir, a.target)
        os.makedirs(dest_dir, exist_ok=True)
        config = TrainerConfig.from_file(J(a.model_dir, 'config.json'))
        unique_code = config.unique_code()

        df = pd.read_excel(J(a.model_dir, f'validate_{a.target}.xlsx'))
        df_report = pd.read_excel(J(a.model_dir, a.target, 'report.xlsx'), sheet_name='images')

        # caseごとにcountの数だけoriginalにdrawする
        # 間違えている場合は別にdrawする

        selected_rows = []
        for name, rows in df_report.groupby('name'):
            selected_rows.append(rows.iloc[:a.count])
        selected_rows.append(df_report[df_report['correct'] < 1])
        selected_rows = pd.concat(selected_rows)

        data = []
        tq = tqdm(selected_rows.iterrows(), total=len(selected_rows))
        for idx, row in tq:
            diag_org, diag = row[['diag_org', 'gt']]
            image_name = row['image_name']
            rows = df[df['original'] == image_name]
            row_images = []
            for _y, cols in rows.groupby('y'):
                row_image_list = []
                for _x, row in cols.iterrows():
                    # print(item['x'], item['y'], item['pred'])
                    name, filename = row[['name', 'filename']]
                    # i = Image.new('RGBA', (row['width'], row['height']), color=(0,0,0,0,))
                    i = Image.open(f'cache/{config.source}/{diag_org}/{name}/{filename}')
                    pred = np.array([row[c] for i, c in enumerate(unique_code)])
                    draw_frame(i, pred, unique_code)
                    row_image_list.append(np.array(i))

                row_image = cv2.hconcat(row_image_list)
                row_images.append(row_image)

            merged_image = Image.fromarray(cv2.vconcat(row_images))
            # original_image = Image.open(
            #     f'data/images/enda3/{diag_org}/{image_name}.jpg',
            # ).convert('RGBA')
            # original_image = Image.alpha_composite(original_image, overlay)
            d = J(dest_dir, diag)
            os.makedirs(d, exist_ok=True)
            # original_image = original_image.convert('RGB')
            merged_image.save(J(d, f'{image_name}'))
            merged_image.close()

            tq.set_description(f'Drew {diag} {image_name}')
            tq.refresh()


    class CalcResultsArgs(CommonArgs):
        model_dir: str = Field(..., l='--model-dir', s='-d')
        target: str = Field('test', s='-t')

    def run_calc_results(self, a: CalcResultsArgs):
        dest_dir= J(a.model_dir, a.target)
        os.makedirs(dest_dir, exist_ok=True)
        config = TrainerConfig.from_file(J(a.model_dir, 'config.json'))
        unique_code = config.unique_code()
        df = pd.read_excel(J(a.model_dir, f'validate_{a.target}.xlsx'))

        data_by_case = []
        for name, items in tqdm(df.groupby('name')):
            diag_org, diag = items.iloc[0][['diag_org', 'diag']]
            preds = items[unique_code]

            preds_sum = np.sum(preds, axis=0)
            preds_sum = preds_sum / np.sum(preds_sum)
            pred_sum = unique_code[np.argmax(preds_sum)]

            preds_label = np.argmax(preds, axis=1)

            unique_values, counts = np.unique(preds_label, return_counts=True)
            pred_vote = unique_code[unique_values[np.argmax(counts)]]

            d = {
                'name': name,
                'diag_org': diag_org,
                'gt': diag,
                'pred(vote)': pred_vote,
                'pred(sum)': pred_sum,
                'correct': int(diag == pred_sum),
            }
            for p, code in zip(preds_sum, unique_code):
                d[code] = p
            data_by_case.append(d)

        data_by_image = []
        for image_name, items in tqdm(df.groupby('original')):
            diag_org, diag, name = items.iloc[0][['diag_org', 'diag', 'name']]

            preds = items[unique_code]
            preds_sum = np.sum(preds, axis=0)
            pred_sum = unique_code[np.argmax(preds_sum)]

            preds_label = np.argmax(preds, axis=1)

            unique_values, counts = np.unique(preds_label, return_counts=True)
            pred_vote = unique_code[unique_values[np.argmax(counts)]]

            d = {
                'name': name,
                'image_name': image_name,
                'diag_org': diag_org,
                'gt': diag,
                'pred(vote)': pred_vote,
                'pred(sum)': pred_sum,
                'correct': int(diag == pred_sum),
            }
            for p, code in zip(preds_sum, unique_code):
                d[code] = p
            data_by_image.append(d)

        df_by_case = pd.DataFrame(data_by_case).sort_values(['diag_org'])
        df_by_image = pd.DataFrame(data_by_image).sort_values(['diag_org', 'image_name'])

        for df, t in ((df_by_case, 'case'), (df_by_image, 'image')):
            for code in unique_code:
                p = df_by_case[code]
                gt = df_by_case['gt'] == code
                fpr, tpr, __t = skmetrics.roc_curve(gt, p)
                auc = skmetrics.auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{code}: AUC={auc:.3f}')
                plt.savefig(J(dest_dir, f'{t}_{code}.png'))
                plt.legend()
                plt.close()
                print(t, code, auc)

        with pd.ExcelWriter(with_wrote(J(dest_dir, 'report.xlsx')), engine='xlsxwriter') as writer:
            df_by_case.to_excel(writer, sheet_name='cases', index=False)
            df_by_image.to_excel(writer, sheet_name='images', index=False)

    class ClusterArgs(CommonArgs):
        file: str
        dest: str = ''
        count: int = 30
        noshow: bool = False
        out: str = ''
        n_neighbors:int = 15
        min_dist:float = 0.1
        no_leiden: bool = False
        unique_code: str = 'LMGAOB'

    def run_cluster(self, a):
        from umap import UMAP

        dest = a.dest or os.path.dirname(a.file)
        unique_codes = a.unique_code or list(df['diag_org'].unique())
        data = torch.load(a.file)

        # results = []
        # features = []
        # for row in data:
        #     features.append(row['feature'])
        #     del row['feature']
        #     results.append(row)

        df = pd.DataFrame(data)
        # features = np.stack(features)

        # df = pd.DataFrame(torch.load(a.file))
        # limit count along name
        rowss = []
        for name, _rows in df.groupby('name'):
            rows = df.loc[np.random.choice(_rows.index, a.count)]
            rowss.append(rows)
        df = pd.concat(rowss)

        features = np.stack(df['feature'])
        labels = df['diag_org'].values

        umap_model = UMAP(
            n_neighbors=a.n_neighbors,
            min_dist=a.min_dist,
            n_jobs=1,
            random_state=42,
        )
        print('Start projection')
        embedding = umap_model.fit_transform(features)
        print('Done projection')
        embedding_x = embedding[:, 0]
        embedding_y = embedding[:, 1]

        knn_graph = umap_model.graph_

        # グラフをiGraph形式に変換
        edges = list(zip(sources, targets))
        graph = ig.Graph(edges=edges, directed=False)
        graph.es['weight'] = weights
        sources, targets = knn_graph.nonzero()
        weights = knn_graph.data

        # Leidenクラスタリングの実行
        partition = la.find_partition(graph, la.RBConfigurationVertexPartition, weights=weights)
        clusters = np.array(partition.membership)

        plots = pd.DataFrame({
            'UMAP1': embedding[:, 0],
            'UMAP2': embedding[:, 1],
            'Diagnosis': labels,
            'Cluster': clusters
        })

        images = []
        for i, row in df.iterrows():
            p = J(os.path.expanduser('~/.cache/endaaman/bt/tiles/enda4_512'),
                  row['diag_org'], row['name'], row['filename'])
            images.append(Image.open(p).copy())

        fig = plt.figure(figsize=(15, 12))
        ax0 = fig.add_subplot(1, 1 if a.no_leiden else 2, 1)
        scatter0 = sns.scatterplot(data=plots, x='UMAP1', y='UMAP2',
                                   hue='Diagnosis', hue_order=unique_codes,
                                   palette='tab10', s=10, ax=ax0)
        hover_images_on_scatters([scatter0.collections[0]], [images], ax=ax0)

        if not a.no_leiden:
            ax1 = fig.add_subplot(1, 2, 2)
            ax0.set_zorder(2)
            ax1.set_zorder(1)
            scatter1 = sns.scatterplot(data=plots, x='UMAP1', y='UMAP2',
                                       hue='Cluster',
                                       palette='Set2', s=10, ax=ax1)
            hover_images_on_scatters([scatter1.collections[0]], [images], ax=ax1)

        dir = os.path.dirname(a.file)
        plt.savefig(J(dir, a.out or f'umap_{a.count}_{a.n_neighbors}_{a.min_dist}.png'))
        if not a.noshow:
            plt.show()


    def run_list_reports(self, a):
        df = []
        for fold in range(5):
            for model_dir in glob(f'out/enda3_512/LMGGGB/fold5_{fold}/*'):
                name = os.path.split(model_dir)[-1]
                fp = J(model_dir, 'test/report.xlsx')
                if not os.path.exists(fp):
                    continue
                df_case = pd.read_excel(fp, sheet_name='cases')
                df_image = pd.read_excel(fp, sheet_name='images')
                df.append({
                    'name': name,
                    'fold': fold,
                    'case': np.mean(df_case['correct']),
                    'image': np.mean(df_image['correct'])
                })
        df = pd.DataFrame(df)
        df.to_excel(with_wrote('out/enda3_512/LMGGGB/acc.xlsx'))


    class GatherReportsArgs(CommonArgs):
        model:str
        target:str = 'test'

    def run_gather_reports(self, a):
        df_cases = []
        df_images = []
        for fold in range(5):
            for model_dir in glob(f'out/enda3_512/LMGGGB/fold5_{fold}/{a.model}'):
                fp = J(model_dir, a.target, 'report.xlsx')
                if not os.path.exists(fp):
                    continue
                df_case = pd.read_excel(fp, sheet_name='cases')
                df_image = pd.read_excel(fp, sheet_name='images')
                df_case['fold'] = fold
                df_image['fold'] = fold
                df_cases.append(df_case)
                df_images.append(df_image)
        # df.to_excel('out/enda3_512/LMGGGB/acc.xlsx')

        dest_dir = f'out/enda3_512/LMGGGB/report_{a.model}'
        os.makedirs(dest_dir, exist_ok=True)

        df_cases = pd.concat(df_cases, ignore_index=True)
        df_images = pd.concat(df_images, ignore_index=True)

        df_images = df_images[['fold', *df_images.columns[df_images.columns!='fold']]]
        df_cases = df_cases[['fold', *df_cases.columns[df_cases.columns!='fold']]]

        unique_code = [c for c in df_cases.columns if c in 'LMGAOB']

        df_case_metrics = []
        df_image_metrics = []
        for df_base, df_out in [[df_cases, df_case_metrics], [df_images, df_image_metrics]]:
            for fold in range(-1, 5):
                if fold < 0:
                    df = df_base
                else:
                    df = df_base[df_base['fold'] == fold]

                acc = np.mean(df['correct'])

                d = {
                    'fold': fold,
                    'acc': acc,
                }

                # precision
                gt = df['gt'].map(lambda x: unique_code.index(x))
                p = df['pred(sum)'].map(lambda x: unique_code.index(x))
                precisions = skmetrics.precision_score(gt, p, average=None)

                for code in unique_code:
                    p = df[code]
                    gt = df['gt'] == code
                    fpr, tpr, __t = skmetrics.roc_curve(gt, p)
                    auc = skmetrics.auc(fpr, tpr)
                    if fold < 0:
                        plt.plot(fpr, tpr, label=f'{code}: AUC={auc:.3f}')
                        plt.legend(loc='lower right')
                        plt.savefig(J(dest_dir, f'roc_all_{code}.png'))
                        plt.close()
                    d[f'auroc_{code}'] = auc

                # ROCs
                for code in unique_code:
                    p = df[code]
                    gt = df['gt'] == code
                    fpr, tpr, __t = skmetrics.roc_curve(gt, p)
                    auc = skmetrics.auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f'{code}: AUC={auc:.3f}')
                    plt.legend(loc='lower right')
                    plt.savefig(J(dest_dir, f'roc_all_{code}.png'))
                    plt.close()


                d['precision'] = np.mean(precisions)
                for code, precision in zip(unique_code, precisions):
                    d[f'precision_{code}'] = precision
                df_out.append(d)

        df_case_metrics = pd.DataFrame(df_case_metrics)
        df_image_metrics = pd.DataFrame(df_image_metrics)

        with pd.ExcelWriter(with_wrote(J(dest_dir, 'report.xlsx')), engine='xlsxwriter') as writer:
            df_case_metrics.to_excel(writer, sheet_name='case_metric', index=False)
            df_image_metrics.to_excel(writer, sheet_name='image_metric', index=False)
            df_cases.to_excel(writer, sheet_name='cases')
            df_images.to_excel(writer, sheet_name='images')


        data = {code: [] for code in unique_code}

        for code in unique_code:
            fprs = []
            tprs = []
            for fold in range(5):
                df = df_cases[df_cases['fold'] == fold]
                p = df[code]
                gt = df['gt'] == code
                fpr, tpr, __t = skmetrics.roc_curve(gt, p)
                fprs.append(fpr)
                tprs.append(tpr)

            self.draw_curve_with_ci(fprs, tprs)
            plt.legend(loc='lower right')
            plt.savefig(J(dest_dir, f'roc_ci_{code}.png'))
            plt.close()

        plt.show()


    def draw_curve_with_ci(self, xx, yy, fill=True, label='{}', color=['blue', 'lightblue'], std_scale=2):
        l = []
        mean_x = np.linspace(0, 1, 1000)
        aucs = []
        for (x, y) in zip(xx, yy):
            l.append(np.interp(mean_x, x, y))
            aucs.append(skmetrics.auc(x, y))

        yy = np.array(l)
        mean_y = yy.mean(axis=0)
        std_y = yy.std(axis=0)

        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        auc_label = f'AUC:{mean_auc:0.3f} ± {std_auc:0.3f}'
        plt.plot(mean_x, mean_y, label=label.format(auc_label), color=color[0])

        if fill:
            plt.fill_between(
                mean_x,
                (mean_y - std_scale * std_y).clip(.0, 1.0),
                (mean_y + std_scale * std_y).clip(.0, 1.0),
                color=color[1], alpha=0.2, label='± 1.0 s.d.')


    class GridAndSamplesArgs(CommonArgs):
        path: str
        size: int = 512
        dest: str = 'out/figs/fig1/'
        with_tiles: bool = False
        width: 3200
        height: 2000

    def run_grid_and_samples(self, a:GridAndSamplesArgs):
        path = a.path
        if not os.path.exists(path):
            path = {
                'G': 'data/images/enda4/G/19-1132_30.jpg',
                'M': 'data/images/enda4/M/N20-205_25.jpg',
                'L': 'data/images/enda4/L/20-3557_09.jpg',
            }[a.path]

        org_img = Image.open(path).crop((0, 0, a.width, a.height))
        img = org_img.copy()
        draw = ImageDraw.Draw(img)

        hor_sizes = n_split(img.width, max(img.width//512, 1))
        ver_sizes = n_split(img.height, max(img.height//512, 1))

        # Draw horizontal lines
        y = 0
        for h in ver_sizes:
            if y > 0:
                draw.line([(0, y), (img.width, y)], fill='red', width=4)
            y += h

        x = 0
        for w in hor_sizes:
            if x > 0:
                draw.line([(x, 0), (x, img.height)], fill='red', width=4)
            x += w

        name = os.path.splitext(os.path.basename(path))[0]
        img = img.crop((0, 0, x, y)).resize((x//4, y//4))
        img.save(with_wrote(J(a.dest, f'grid_{name}.png')))

        if a.with_tiles:
            org_img = org_img.crop((0, 0, x, y))
            tiles = grid_split(org_img, 512, overwrap=False, flattern=False)
            # flatten
            tiles = sum(tiles, [])
            for i, tile in enumerate(tiles):
                tile.save(with_mkdir(J(a.dest, f'tiles_{name}', f'{i}.png')))





if __name__ == '__main__':
    cli = CLI()
    cli.run()
