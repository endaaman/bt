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
from endaaman.trainer import Trainer, TrainCommander
from endaaman.metrics import MultiAccuracy

from models import create_model, CrossEntropyLoss, NestedCrossEntropyLoss
from datasets import BTDataset



class T(Trainer):
    def prepare(self):
        self.criterion = CrossEntropyLoss()

    def eval(self, inputs, labels):
        outputs = self.model(inputs.to(self.device))
        loss = self.criterion(outputs, labels.to(self.device))
        return loss, outputs

    def create_optimizer(self, lr):
        return optim.RAdam(self.model.parameters(), lr=lr)

    def create_scheduler(self, lr):
        # return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 0.95 ** x)
        return CosineLRScheduler(
            self.optimizer, t_initial=100, lr_min=0.00001,
            warmup_t=50, warmup_lr_init=0.00005, warmup_prefix=True)

    def hook_load_state(self, checkpoint):
        self.scheduler.step(checkpoint.epoch-1)

    def step(self, train_loss):
        self.scheduler.step(self.current_epoch)

    def get_batch_metrics(self):
        return {
            'acc': MultiAccuracy(),
        }

    def get_epoch_metrics(self):
        return { }


class C(TrainCommander):
    def arg_common(self, parser):
        parser.add_argument('--size', '-s', type=int, default=768)

    def create_loaders(self, num_classes):
        return [self.as_loader(BTDataset(
            target=t,
            crop_size=self.args.size,
            size=self.args.size,
            seed=self.args.seed,
        )) for t in ['train', 'test']]

    def arg_start(self, parser):
        parser.add_argument('--model', '-m', default='tf_efficientnetv2_b0_3')

    def run_start(self):
        model = create_model(self.args.model).to(self.device)
        loaders = self.create_loaders(model.num_classes)

        trainer = self.create_trainer(
            T=T,
            name=self.args.model,
            model=model,
            loaders=loaders,
        )

        trainer.start(self.args.epoch, lr=self.args.lr)

    def arg_resume(self, parser):
        parser.add_argument('--checkpoint', '-c', required=True)

    def run_resume(self):
        checkpoint = torch.load(self.args.checkpoint)
        model = create_model(checkpoint.name).to(self.device)
        loaders = self.create_loaders(model.num_classes)

        trainer = self.create_trainer(
            T=T,
            name=checkpoint.name,
            model=model,
            loaders=loaders,
        )

        trainer.start(self.args.epoch, checkpoint=checkpoint)


if __name__ == '__main__':
    c = C({
        'epoch': 200,
        'lr': 0.0001,
        'batch_size': 16,
        'save_period': 50,
    })
    c.run()
