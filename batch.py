import os
import re
from glob import glob
import itertools
import hashlib
from itertools import compress

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
from torchvision import transforms
from sklearn import metrics as skmetrics
import umap

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from endaaman import load_images_from_dir_or_file, with_wrote, grid_split, with_mkdir
from endaaman.ml import BaseMLCLI

from datasets.utils import show_fold_diag
from utils import calc_white_area
from classification import TrainerConfig


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
        model_dir: str = Field(..., cli=('--model-dir', '-d'))
        target: str = 'test'

    def run_draw_result(self, a):
        dest_dir= J(a.model_dir, a.target)
        os.makedirs(dest_dir, exist_ok=True)
        config = TrainerConfig.from_file(J(a.model_dir, 'config.json'))
        unique_code = config.unique_code()

        df = pd.read_excel(J(a.model_dir, f'{a.target}.xlsx'))
        df['image_name'] = df['name'].str.cat(df['order'].astype(str), sep='_')

        colors = {
            'L': 'green',
            'M': 'blue',
            'G': 'red',
            'A': 'yellow',
            'O': 'purple',
            'B': 'black',
        }

        font = ImageFont.truetype('/usr/share/fonts/ubuntu/Ubuntu-R.ttf', size=16)

        data = []
        tq = tqdm(df.groupby('image_name'))
        for image_name, rows in tq:
            diag_org, diag = rows.iloc[0][['diag_org', 'diag']]

            row_images = []
            for _y, cols in rows.groupby('y'):
                row_image_list = []
                for _x, item in cols.iterrows():
                    # print(item['x'], item['y'], item['pred'])
                    name, filename = item[['name', 'filename']]
                    # i = Image.new('RGBA', (item['width'], item['height']), color=(0,0,0,0,))
                    i = Image.open(f'cache/{config.source}/{diag_org}/{name}/{filename}')
                    draw = ImageDraw.Draw(i)
                    draw.rectangle(
                        xy=((0, 0), (item['width'], item['height'])),
                        outline=colors[item['pred']],
                    )
                    text = ' '.join([f'{k}:{item[k]:.2f}' for k in unique_code])
                    bb = draw.textbbox(xy=(0, 0), text=text, font=font, spacing=8)
                    draw.rectangle(
                        xy=bb,
                        fill=colors[item['pred']],
                    )
                    draw.text(
                        xy=(0, 0),
                        text=text,
                        font=font,
                        fill='white'
                    )
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
            merged_image.save(J(d, f'{image_name}.jpg'))
            merged_image.close()

            tq.set_description(f'Drew {image_name}')
            tq.refresh()



    class CalcResultArgs(CommonArgs):
        model_dir: str = Field(..., cli=('--model-dir', '-d'))
        target: str = 'test'

    def run_calc_result(self, a):
        dest_dir= J(a.model_dir, a.target)
        os.makedirs(dest_dir, exist_ok=True)
        config = TrainerConfig.from_file(J(a.model_dir, 'config.json'))
        unique_code = config.unique_code()
        df = pd.read_excel(J(a.model_dir, f'{a.target}.xlsx'))

        df['image_name'] = df['name'].str.cat(df['order'].astype(str), sep='_')

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
        for image_name, items in tqdm(df.groupby('image_name')):
            diag_org, diag = items.iloc[0][['diag_org', 'diag']]

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

        print('Acc by case')
        print(df_by_case['correct'].mean())
        print('Acc by image')
        print(df_by_image['correct'].mean())

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
        model_dir: str = Field(..., cli=('--model-dir', '-d'))
        target: str = 'test'
        count: int = 10
        show: bool = Field(False, cli=('--show', ))
        mode: str = Field('all', regex=r'uniq|all')
        notrained: bool = Field(False, cli=('--notrained', ))

    def run_cluster(self, a):
        config = TrainerConfig.from_file(J(a.model_dir, 'config.json'))

        unique_code = config.unique_code()

        df = pd.read_excel(J(a.model_dir, f'{a.target}.xlsx'), index_col=0)
        features = dict(torch.load(J(a.model_dir, f'{a.target}_features.pt')))

        df = pd.read_excel('cache/enda3_512/tiles.xlsx')
        df['diag_org'] = df['diag']
        features = {}
        for f in range(5):
            f = dict(torch.load(f'out/enda3_512/LMGGGB/fold5_{f}/resnetrs50/test_features.pt'))
            features.update(f)

        target_features = []
        diags = []
        images = []

        center_crop = transforms.CenterCrop((512, 512))

        for name, rows in df.groupby('name'):
            selected_rows = df.loc[np.random.choice(rows.index, a.count)]
            # 19-3046_1_0_0
            selected_features = [
                features['_'.join(str(s) for s in [name, row['order'], row['x'], row['y']])]
                for i, row in selected_rows.iterrows()
            ]

            # print(selected_rows)
            # print(selected_features[0].shape)
            # break
            target_features += selected_features
            if a.mode == 'uniq':
                diags += [unique_code.index(d) for d in selected_rows['diag']]
            else:
                diags += ['LMGAOB'.index(d) for d in selected_rows['diag_org']]

            images += [
                np.array(center_crop(Image.open(f'cache/{config.source}/{diag}/{name}/{fn}')))
                for _, (diag, name, fn) in selected_rows[['diag_org', 'name', 'filename']].iterrows()
            ]

        diags = np.array(diags)
        print('Loaded samples.')

        ##* UMAP
        embedding = umap.UMAP().fit_transform(target_features)
        embedding_x = embedding[:, 0]
        embedding_y = embedding[:, 1]

        ##* Plot
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))

        target_features = np.array(target_features)
        scs = []
        dds = []
        iis = []
        for n in np.unique(diags):
            label = 'LMGAOB'[n] if a.mode == 'uniq' else 'LMGAOB'[n]
            scs.append(plt.scatter(embedding_x[diags == n], embedding_y[diags == n], label=label, s=24))
            dds.append(diags[diags == n])
            iis.append([i for (i, d) in zip(images, diags) if d == n])

        # annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
        #                 bbox=dict(boxstyle="round", fc="w"),
        #                 arrowprops=dict(arrowstyle="->"))
        imagebox = OffsetImage(images[0], zoom=.5)
        imagebox.image.axes = ax

        annot = AnnotationBbox(imagebox,
                               xy=(0, 0),
                               # xybox=(256, 256),
                               # xycoords='data',
                               boxcoords='offset points',
                               # boxcoords=('axes fraction', 'data'),
                               pad=0.1,
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3'))
        annot.set_visible(False)
        ax.add_artist(annot)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes != ax:
                return

            for n, (sc, ii) in enumerate(zip(scs, iis)):
                cont, index = sc.contains(event)
                if cont:
                    i = index['ind'][0]
                    pos = sc.get_offsets()[i]
                    annot.xy = pos
                    annot.xybox = pos + np.array([150, 30])
                    image = ii[i]
                    # text = unique_code[n]
                    # annot.set_text(text)
                    # annot.get_bbox_patch().set_facecolor(cmap(int(text)/10))
                    imagebox.set_data(image)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return

            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()
                return

        fig.canvas.mpl_connect('motion_notify_event', hover)

        plt.legend()
        d = J(a.model_dir, 'umap')
        os.makedirs(d, exist_ok=True)
        plt.savefig(with_wrote(J(d, f'{a.target}_{a.mode}_{a.count}.png')))
        if a.show:
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

        dest_dir = with_mkdir(f'out/enda3_512/LMGGGB/report_{a.model}')

        df_cases = pd.concat(df_cases, ignore_index=True)
        df_images = pd.concat(df_images, ignore_index=True)

        df_images = df_images[['fold', *df_images.columns[df_images.columns!='fold']]]
        df_cases = df_cases[['fold', *df_cases.columns[df_cases.columns!='fold']]]

        unique_code = [c for c in df_cases.columns if c in 'LMGAOB']

        df_metrics = []
        for fold in range(-1, 5):
            if fold < 0:
                df = df_cases
            else:
                df = df_cases[df_cases['fold'] == fold]

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
            df_metrics.append(d)

        df_metrics = pd.DataFrame(df_metrics)

        with pd.ExcelWriter(with_wrote(J(dest_dir, 'report.xlsx')), engine='xlsxwriter') as writer:
            df_metrics.to_excel(writer, sheet_name='metrics', index=False)
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



if __name__ == '__main__':
    cli = CLI()
    cli.run()
