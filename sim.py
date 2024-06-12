import os
import re
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torchvision import transforms
from pydantic import Field
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from endaaman.ml import BaseTrainer, BaseTrainerConfig, BaseMLCLI, Checkpoint

from models import SimSiamModel, TimmModel
from loss import SymmetricCosSimLoss
from datasets import MEAN, STD
from datasets.fold import PairedFoldDataset, FoldDataset


J = os.path.join


class TrainerConfig(BaseTrainerConfig):
    model_name:str = 'resnetrs50'
    source: str = 'enda4_512'
    code: str = 'LMGAOB'
    total_fold: int = 5
    fold: int = 0
    size: int = 256
    minimum_area: float = 0.6
    limit: int = 100
    noupsample: bool = False

    mean: float = MEAN
    std: float = STD

    def unique_code(self):
        return [c for c in dict.fromkeys(self.code) if c in 'LMGAOB']

    def num_classes(self):
        return len(self.unique_code())


class Trainer(BaseTrainer):
    def prepare(self):
        model = SimSiamModel(name=self.config.model_name)
        self.criterion = SymmetricCosSimLoss()
        return model

    def create_optimizer(self):
        return optim.RAdam(self.model.parameters())

    def eval(self, inputs, __gts, i):
        inputs = inputs.to(self.device)
        z0, z1, p0, p1 = self.model(inputs)
        loss = self.criterion(z0, z1, p0, p1)
        return loss, None

    # def metrics_precision(self, preds, gts, batch):
    #     if batch:
    #         return None
    #     preds = preds.detach().cpu()
    #     labels = torch.unique(preds)
    #     correct = 0
    #     for label in labels:
    #         items = gts[preds == label]
    #         elements, counts = torch.unique(items, return_counts=True)
    #         dominant = elements[torch.argmax(counts)]
    #         # print(label, dominant, torch.sum(items == dominant))
    #         correct += torch.sum(items == dominant)
    #     return correct/len(preds)


class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        pass

    class TrainArgs(CommonArgs, TrainerConfig):
        base_lr: float = 1e-5 # ViT for 1e-6
        lr: float = -1
        batch_size: int = Field(48, s='-B')
        num_workers: int = Field(4, s='-N')
        epoch: int = Field(30, s='-E')
        suffix: str = ''
        out: str = 'out/{source}/SimSiam/fold{total_fold}_{fold}/{model_name}{suffix}'
        overwrite: bool = Field(False, s='-O')


    def run_train(self, a:TrainArgs):
        m = re.match(r'^.*_(\d+)$', a.source)
        assert m
        a.lr = a.lr if a.lr>0 else a.base_lr*a.batch_size

        config = TrainerConfig(**a.dict())

        dss = [PairedFoldDataset(
             source_dir = J('data/tiles', a.source),
             total_fold = a.total_fold,
             fold = a.fold,
             target = t,
             code = a.code,
             size = a.size,
             minimum_area = a.minimum_area,
             limit = a.limit,
             upsample = not config.noupsample,
             augmentation = True,
             normalization = True,
        ) for t in ('train', 'test')]

        out_dir = a.out.format(**a.dict())

        trainer = Trainer(
            config=config,
            out_dir=out_dir,
            train_dataset=dss[0],
            val_dataset=dss[1],
            use_gpu=True,
            multi_gpu=True,
            overwrite=a.overwrite,
            experiment_name=a.source,
        )

        trainer.start(a.epoch)


    class ValidateArgs(CommonArgs):
        model_dir: str = Field(..., s='-d')
        target: str = Field('test', choices=['train', 'test', 'all'])
        batch_size: int = Field(128, s='-B')
        use_best: bool = False

    def run_validate(self, a:ValidateArgs):
        chp = 'checkpoint_best.pt' if a.use_best else 'checkpoint_last.pt'
        checkpoint = Checkpoint.from_file(J(a.model_dir, chp))
        config = TrainerConfig.from_file(J(a.model_dir, 'config.json'))
        model = SimSiamModel(name=config.model_name)
        print('config:', config)
        model.load_state_dict(checkpoint.model_state)
        model = model.cuda().eval()

        transform = transforms.Compose([
            transforms.CenterCrop(config.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

        ds = FoldDataset(
             total_fold=config.total_fold,
             fold=config.fold,
             source_dir=J('data/tiles', config.source),
             target=a.target,
             code=config.code,
             size=config.size,
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
            with torch.set_grad_enabled(False):
                f = model.forward_features(tt.cuda(), use_mlp=False)
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
        torch.save(data, J(a.model_dir, f'features_{a.target}.pt'))

    class ClusterArgs(CommonArgs):
        count: int = 50

    def run_cluster(self, a):
        from umap import UMAP
        df = pd.DataFrame(torch.load('out/enda4_512/SimSiam/fold5_0/resnetrs50/features_test2.pt'))

        rowss = []
        for name, _rows in df.groupby('name'):
            rows = df.loc[np.random.choice(_rows.index, a.count)]
            rowss.append(rows)
        df = pd.concat(rowss)

        features = np.stack([f.numpy() for f in df['feature']])
        print('features', features.shape)

        labels = df['diag_org'].values

        umap_model = UMAP(n_components=2, min_dist=0.1)
        print('Start projection')
        embedding = umap_model.fit_transform(features)
        print('Done projection')
        embedding_x = embedding[:, 0]
        embedding_y = embedding[:, 1]

        for i, label in enumerate('LMGAOB'):
            needle = labels == label
            xx, yy = embedding_x[needle], embedding_y[needle]
            plt.scatter(xx, yy, label=label, c=f'C{i}')
        plt.show()

        # reducer = umap.UMAP(n_components=2, random_state=42)
        # reduced_features = reducer.fit_transform(features)
        # # DBSCANを使用してクラスタリングの実行
        # dbscan = DBSCAN(eps=0.5, min_samples=5)  # epsとmin_samplesはデータに応じて調整
        # clusters = dbscan.fit_predict(reduced_features)

        # # クラスタリングの評価（クラスタが複数ある場合にのみ計算）
        # if len(set(clusters)) > 1:
        #     silhouette_avg = silhouette_score(reduced_features, clusters)
        #     print(f"Silhouette Score: {silhouette_avg}")
        # else:
        #     print("クラスタ数が1つのみです。Silhouette Scoreを計算できません。")

        # # クラスタリング結果の可視化
        # plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=clusters, cmap='viridis')
        # plt.title('Clustering of Features Extracted by SimSiam (UMAP + DBSCAN)')
        # plt.show()



if __name__ == '__main__':
    cli = CLI()
    cli.run()
