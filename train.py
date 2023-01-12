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
from endaaman.metrics import MultiAccuracy

from models import ModelId, create_model, CrossEntropyLoss, NestedCrossEntropyLoss
from datasets import BTDataset



class MyTrainer(Trainer):
    def prepare(self, **kwargs):
        self.criterion = CrossEntropyLoss(input_logits=True)

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
            warmup_lr_init=lr/2, lr_min=lr/100,
            warmup_prefix=True)

    def hook_load_state(self, checkpoint):
        self.scheduler.step(checkpoint.epoch-1)

    def step(self, train_loss):
        self.scheduler.step(self.current_epoch)


    def get_metrics(self):
        return {
            'batch': {
                'acc': MultiAccuracy(),
            },
            'epoch': { },
        }


class CMD(TorchCommander):
    def arg_common(self, parser):
        parser.add_argument('--crop', '-c', type=int, default=768*2)
        parser.add_argument('--size', '-s', type=int, default=768)

    def create_loaders(self, num_classes):
        return [self.as_loader(BTDataset(
            target=t,
            crop_size=self.args.crop,
            size=self.args.size,
            seed=self.args.seed,
            merge_G=num_classes == 3,
        )) for t in ['train', 'test']]

    def arg_start(self, parser):
        parser.add_argument('--model', '-m', default='tf_efficientnetv2_b0_5')

    def run_start(self):
        model_id = ModelId.from_str(self.a.model)
        assert model_id.num_classes in [3, 5]
        loaders = self.create_loaders(model_id.num_classes)

        trainer = self.create_trainer(
            T=MyTrainer,
            model_name=self.a.model,
            loaders=loaders,
            log_dir='data/logs',
        )

        trainer.start(self.args.epoch, lr=self.args.lr)

    # def arg_resume(self, parser):
    #     parser.add_argument('--checkpoint', '-c', required=True)
    #
    # def run_resume(self):
    #     checkpoint = torch.load(self.args.checkpoint)
    #     model = create_model(checkpoint.name).to(self.device)
    #     loaders = self.create_loaders(model.num_classes)
    #
    #     trainer = self.create_trainer(
    #         T=MyTrainer,
    #         name=checkpoint.name,
    #         model=model,
    #         loaders=loaders,
    #     )
    #
    #     trainer.start(self.args.epoch, checkpoint=checkpoint)


if __name__ == '__main__':
    cmd = CMD({
        'epoch': 200,
        'lr': 0.0001,
        'batch_size': 16,
        'save_period': 50,
    })
    cmd.run()
