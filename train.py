import os
import re
from glob import glob

import numpy as np
import torch
from torch import nn
from torch import optim
# import torch_optimizer as optim2
from torchvision.utils import make_grid
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn import metrics
from timm.scheduler.cosine_lr import CosineLRScheduler
# from endaaman.torch import TrainCommander
from endaaman.torch import Trainer, TorchCommander, pil_to_tensor, tensor_to_pil
from endaaman.metrics import MultiAccuracy, AccuracyByChannel, MetricsFn
from endaaman.functional import multi_accuracy

from models import ModelId, create_model, CrossEntropyLoss, NestedCrossEntropyLoss
from datasets import BrainTumorDataset, NUM_TO_DIAG



def label_to_text(result):
    ss = []
    for i, v in enumerate(result):
        label = NUM_TO_DIAG[i]
        ss.append(f'{label}:{v:.3f}')
    return '\n'.join(ss)


class LMGAccuracy(MetricsFn):
    def __call__(self, preds, gts):
        summed = torch.sum(preds[:, [2, 3, 4]], dim=-1)
        preds[:, 2] = summed
        preds = preds[:, :3]
        gts = gts.clamp(max=2)
        return multi_accuracy(preds, gts, by_index=True)


class MyTrainer(Trainer):
    def prepare(self, **kwargs):
        self.merge_weights = kwargs.pop('merge_weights', [1.0, 0.0])
        self.lr_min = kwargs.pop('lr_min', 10)
        self.lr_plateau = kwargs.pop('lr_plateau', -1)
        self.lr_decay = kwargs.pop('lr_decay', -1)
        assert len(kwargs.items()) == 0

        self.font = ImageFont.truetype('/usr/share/fonts/ubuntu/UbuntuMono-R.ttf', 36)
        self.num_classes = self.model.num_classes

        if self.num_classes == 5:
            self.criterion = NestedCrossEntropyLoss(
                rules=[{
                    'weight': self.merge_weights[0],
                    'index': [],
                }, {
                    'weight': self.merge_weights[1],
                    'index': [[2, 3, 4]], # merge G,A,O to G
                }])
        else:
            self.criterion = CrossEntropyLoss(input_logits=True)

    def create_model(self):
        model_id = ModelId.from_str(self.model_name)
        return create_model(model_id).to(self.device)

    def eval(self, inputs, labels):
        outputs = self.model(inputs.to(self.device), activate=False)
        loss = self.criterion(outputs, labels.to(self.device))
        return loss, outputs

    def eval_image(self, inputs, labels, outputs):
        images = []
        for input_, label, output in zip(inputs, labels, outputs):
            result = torch.softmax(output, dim=0).detach().cpu()
            correct = label.item() == torch.argmax(result, dim=0)
            if correct:
                continue
            image = tensor_to_pil(input_)
            image = image.resize((256, 256))
            draw = ImageDraw.Draw(image)
            draw.text(
                xy=(0, 0),
                text=f'GT: {NUM_TO_DIAG[label.item()]}\n' + label_to_text(result.tolist()),
                fill='navy',
                align='left',
                font=self.font)
            images.append(image)
        return [pil_to_tensor(i) for i in images]

    def merge_images(self, images):
        images = images[:64]
        # return make_grid(torch.stack(images), nrow=int(np.ceil(np.sqrt(len(images)))))
        return make_grid(torch.stack(images), nrow=8)

    def create_scheduler(self, lr, max_epoch):
        if self.lr_plateau > 0 :
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.lr_plateau)
        if self.lr_decay > 0 :
            return optim.lr_scheduler.LambdaLR(self.optimizer, lambda e: e ** self.lr_decay)
        return CosineLRScheduler(
            self.optimizer,
            warmup_t=0, t_initial=max_epoch,
            warmup_lr_init=lr/2, lr_min=lr/self.lr_min,
            warmup_prefix=True)

    def hook_load_state(self, checkpoint):
        self.scheduler.step(checkpoint.epoch-1)

    def step(self, train_loss):
        self.scheduler.step(self.current_epoch)

    def get_metrics(self):
        if self.num_classes == 5:
            m = {
                'acc5': MultiAccuracy(),
                'acc3': LMGAccuracy(),
            }
        else:
            m = {
                'acc3': MultiAccuracy(),
            }

        common_m = {
            f'acc_{l}{NUM_TO_DIAG[l]}': AccuracyByChannel(target_channel=l) for l in range(self.num_classes)
        }
        return {
            'batch': {**m, **common_m},
            'epoch': { },
        }


class CMD(TorchCommander):
    def arg_common(self, parser):
        parser.add_argument('--grid-size', '-g', type=int, default=768)
        parser.add_argument('--crop-size', '-c', type=int, default=768)
        parser.add_argument('--input-size', '-i', type=int, default=768)
        parser.add_argument('--src-dir', '-s', default='data/images')
        parser.add_argument('--merge-weights', '-w', type=str, default='10')
        parser.add_argument('--lr-min', type=int, default=10)
        parser.add_argument('--lr-plateau', type=int, default=-1)
        parser.add_argument('--lr-decay', type=float, default=-1)

    def pre_common(self):
        assert re.match(r'^\d\d$', self.a.merge_weights)
        self.merge_weights = [float(c) for c in self.a.merge_weights]

    def create_loaders(self, num_classes):
        return self.as_loaders(*[
            BrainTumorDataset(
                target=t,
                aug_mode=t,
                src_dir=self.a.src_dir,
                grid_size=self.a.grid_size,
                crop_size=self.a.crop_size,
                input_size=self.a.input_size,
                seed=self.a.seed,
                merge_G=num_classes == 3,
            ) for t in ['train', 'test']])

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
            merge_weights=self.merge_weights,
            lr_min=self.a.lr_min,
            lr_plateau=self.a.lr_plateau,
            lr_decay=self.a.lr_decay,
        )

        trainer.start(self.a.epoch, lr=self.a.lr)

    def arg_resume(self, parser):
        parser.add_argument('--checkpoint', '-p', required=True)

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
        'epoch': 30,
        'lr': 0.0001,
        'batch_size': 8,
        'save_period': 15,
    })
    cmd.run()
