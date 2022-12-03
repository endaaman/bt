import os
import re
from glob import glob

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import pandas as pd
from PIL import Image
from sklearn import metrics
from endaaman.torch import Trainer, TrainCommander
from endaaman.metrics import MultiAccuracy

from models import create_model, available_models
from datasets import BTDataset



class CrossEntropyLoss(nn.Module):
    def __init__(self, eps=1e-24):
        super().__init__()
        self.eps = eps
        self.loss_fn = nn.NLLLoss()
        # self.num_classes = num_classes

    def forward(self, x, y):
        return self.loss_fn((x + self.eps).log(), y)


class T(Trainer):
    def prepare(self):
        self.criterion = CrossEntropyLoss()

    # def get_scheduler(self, optimizer):
    #     return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 0.99 ** x)

    def eval(self, inputs, labels, device):
        outputs = self.model(inputs.to(device))
        loss = self.criterion(outputs, labels.to(device))
        return loss, outputs


    def get_batch_metrics(self):
        return {
            'acc': MultiAccuracy(),
        }

    def get_epoch_metrics(self):
        return { }


class C(TrainCommander):
    def arg_common(self, parser):
        parser.add_argument('--model', '-m', choices=available_models, default=available_models[0])

    def arg_train(self, parser):
        pass

    def run_train(self):
        model = create_model(self.args.model, 3).to(self.device)

        loaders = [self.as_loader(BTDataset(
            target=t,
            crop_size=768,
            size=768,
            scale=5,
        )) for t in ['train', 'test']]

        trainer = T(
            name=self.args.model,
            model=model,
            loaders=loaders,
        )

        trainer.train(self.args.lr, self.args.epoch, device=self.device)


if __name__ == '__main__':
    c = C({
        'epoch': 50,
        'lr': 0.0001,
        'batch_size': 16,
    })
    c.run()