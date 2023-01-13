import os
import re
from glob import glob

import numpy as np
import torch
from torch import nn
from torch import optim
# import torch_optimizer as optim2
from tqdm import tqdm
import pandas as pd
from PIL import Image
from sklearn import metrics
from timm.scheduler.cosine_lr import CosineLRScheduler
# from endaaman.torch import TrainCommander
from endaaman.torch import Trainer, TorchCommander
from endaaman.metrics import MultiAccuracy, AccuracyByChannel

from models import ModelId, create_model, CrossEntropyLoss, NestedCrossEntropyLoss
from datasets import LMGDataset, NUM_TO_DIAG


class MyTrainer(Trainer):
    def prepare(self, **kwargs):
        # self.criterion = NestedCrossEntropyLoss(input_logits=True) if use_nested else CrossEntropyLoss(input_logits=True)
        self.criterion = nn.CrossEntropyLoss()

    def create_model(self):
        model_id = ModelId.from_str(self.model_name)
        return create_model(model_id).to(self.device)

    def eval(self, inputs, labels):
        outputs = self.model(inputs.to(self.device), activate=False)
        loss = self.criterion(outputs, labels.to(self.device))
        return loss, outputs

    def create_scheduler(self, lr):
        return CosineLRScheduler(
            self.optimizer,
            warmup_t=5, t_initial=80,
            warmup_lr_init=lr/2, lr_min=lr/10,
            warmup_prefix=True)

    def hook_load_state(self, checkpoint):
        self.scheduler.step(checkpoint.epoch-1)

    def step(self, train_loss):
        self.scheduler.step(self.current_epoch)

    def get_metrics(self):
        model_id = ModelId.from_str(self.model_name)
        return {
            'batch': {
                'acc': MultiAccuracy(),
                **{
                    f'acc{NUM_TO_DIAG[l]}': AccuracyByChannel(target_channel=l) for l in range(model_id.num_classes)
                }
            },
            'epoch': { },
        }


class CMD(TorchCommander):
    def arg_common(self, parser):
        parser.add_argument('--crop', '-c', type=int, default=768*2)
        parser.add_argument('--size', '-s', type=int, default=768)
        parser.add_argument('--dir', '-d', default='data/images')
        parser.add_argument('--scale', type=float, default=1.0)
        parser.add_argument('--full', action='store_true')
        parser.add_argument('--grid', action='store_true')


    def create_loaders(self, num_classes):
        return [self.as_loader(LMGDataset(
            target='all' if self.a.full and t == 'train' else 'train',
            aug_mode=t,
            base_dir=self.a.dir,
            crop_size=self.a.crop,
            size=self.a.size,
            seed=self.a.seed,
            scale=self.a.scale,
            merge_G=num_classes == 3,
        )) for t in ['train', 'test']]

    def arg_start(self, parser):
        parser.add_argument('--model', '-m', default='tf_efficientnetv2_b0_3')

    def run_start(self):
        model_id = ModelId.from_str(self.a.model)
        assert model_id.num_classes in [3, 5]
        loaders = self.create_loaders(model_id.num_classes)

        trainer = self.create_trainer(
            T=MyTrainer,
            model_name=self.a.model,
            loaders=loaders,
        )

        trainer.start(self.a.epoch, lr=self.a.lr)

    def arg_resume(self, parser):
        parser.add_argument('--weight', '-w', required=True)

    def run_resume(self):
        checkpoint = torch.load(self.a.checkpoint)

        model_id = ModelId.from_str(checkpoint.model_name)
        assert model_id.num_classes in [3, 5]
        loaders = self.create_loaders(model_id.num_classes)

        trainer = self.create_trainer(
            T=MyTrainer,
            model_name=checkpoint.model_name,
            loaders=loaders,
        )

        trainer.start(self.a.epoch, lr=self.a.lr, checkpoint=checkpoint)


if __name__ == '__main__':
    cmd = CMD({
        'epoch': 200,
        'lr': 0.0001,
        'batch_size': 16,
        'save_period': 50,
    })
    cmd.run()
