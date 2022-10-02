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

from models import create_model
from datasets import BTDataset

from endaaman.torch import Trainer


class CrossEntropyLoss(nn.Module):
    def __init__(self, eps=1e-24):
        super().__init__()
        self.eps = eps
        self.loss_fn = nn.NLLLoss()
        # self.num_classes = num_classes

    def forward(self, x, y):
        return self.loss_fn((x + self.eps).log(), y)

def acc_fn(outputs, labels):
    y_pred = outputs.cpu().argmax(dim=1).detach().numpy()
    y_true = labels.detach().numpy()
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)

# def binary_auc_fn(outputs, labels):
#     y_true = labels.cpu().flatten().detach().numpy()
#     y_pred = outputs.cpu().flatten().detach().numpy()
#     return metrics.roc_auc_score(y_true, y_pred)

available_models = \
    [f'eff_b{i}' for i in range(6)] + \
    [f'vgg{i}' for i in [11, 13, 16, 19]] + \
    [f'vgg{i}_bn' for i in [11, 13, 16, 19]]

class C(Trainer):
    def arg_common(self, parser):
        parser.add_argument('--model', '-m', choices=available_models, default=available_models[0])

    def arg_train(self, parser):
        pass

    def run_train(self):
        model = create_model(self.args.model, 3).to(self.device)

        train_loader, test_loader = [self.as_loader(BTDataset(
            target=t,
            size=512,
            scale=2,
        )) for t in ['train', 'test']]

        criterion = CrossEntropyLoss()

        def eval_fn(inputs, labels):
            outputs = model(inputs.to(self.device))
            loss = criterion(outputs, labels.to(self.device))
            return loss, outputs

        self.train_model(
            self.args.model,
            model,
            train_loader,
            test_loader,
            eval_fn, {
                'acc': acc_fn,
            }, {
                # 'auc': binary_auc_fn,
            })

if __name__ == '__main__':
    c = C({
        'epoch': 50,
        'lr': 0.001,
        'batch_size': 128,
    })
    c.run()
