import os
import math
import json
from glob import glob

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics as skmetrics
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pydantic import Field
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


from datasets.fold import FoldDataset, MEAN, STD
from endaaman.ml.cli import BaseMLCLI, BaseDLArgs, BaseTrainArgs


J = os.path.join


class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        pass

    class ValidateArgs(CommonArgs):
        model:str = 'uni'
        total_fold:int = 5
        fold: int = 0
        source: str = 'enda4_512'
        batch_size:int = 256
        size: int = 512

    def run_validate(self, a):
        if a.model == 'uni':
            model = timm.create_model('hf-hub:MahmoodLab/uni',
                                      pretrained=True,
                                      init_values=1e-5,
                                      dynamic_img_size=True)
            # transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
            cfg = resolve_data_config(model.pretrained_cfg, model=model)
            mean, std = cfg['mean'], cfg['std']
        elif a.model == 'gigapath':
            model = timm.create_model('hf_hub:prov-gigapath/prov-gigapath', pretrained=True)
            cfg = resolve_data_config(model.pretrained_cfg, model=model)
            mean, std = cfg['mean'], cfg['std']
        elif a.model == 'imagenet':
            model = timm.create_model('resnetrs50', pretrained=True)
            model.fc = nn.Identity()
            cfg = resolve_data_config(model.pretrained_cfg, model=model)
            mean, std = cfg['mean'], cfg['std']
        else:
            raise RuntimeError('Invalid model:', a.model)

        transform = transforms.Compose([
            transforms.CenterCrop((256, 256)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        eval_fn = lambda model, t: model(t)
        model = model.cuda().eval()
        out_dir = J('out', 'gpm', a.model)
        os.makedirs(out_dir, exist_ok=True)

        ds = FoldDataset(
             total_fold=a.total_fold,
             fold=a.fold,
             source=a.source,
             target='test',
             code='LMGAOB',
             size=a.size,
             minimum_area=-1,
             augmentation=False,
             normalization=True,
             limit=-1,
        )
        df = ds.df

        num_chunks = math.ceil(len(ds.df) / a.batch_size)
        tq = tqdm(range(num_chunks))
        featuress = []
        for chunk in tq:
            i0 = chunk*a.batch_size
            i1 = (chunk+1)*a.batch_size
            rows = df[i0:i1]
            tt = []
            for i, row in rows.iterrows():
                image = ds.load_from_row(row)
                tt.append(transform(image))
                image.close()

            tt = torch.stack(tt)
            with torch.inference_mode():
                f = eval_fn(model, tt.cuda())
            features = f.detach().cpu()
            featuress.append(features)
            tq.set_description(f'{i0} - {i1}')
            tq.refresh()

        features = torch.cat(featuress)
        features = features.reshape(features.shape[0], features.shape[1])
        # torch.save(features, J(a.model_dir, f'features_{a.target}.pt'))

        data = [
            dict(zip(['name', 'filename', 'diag', 'diag_org', 'feature'], values))
            for values in zip(
                df['name'],
                df['filename'],
                df['diag'],
                df['diag_org'],
                features.numpy()
            )
        ]
        os.makedirs(out_dir, exist_ok=True)
        torch.save(data, J(out_dir, f'features_{a.total_fold}_{a.fold}.pt'))



if __name__ == '__main__':
    cli = CLI()
    cli.run()
