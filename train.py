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
from endaaman.metrics import MultiAccuracy, AccuracyByChannel

from models import ModelId, create_model, CrossEntropyLoss, NestedCrossEntropyLoss
from datasets import LMGDataset, NUM_TO_DIAG



def label_to_text(result):
    ss = []
    for i, v in enumerate(result):
        label = NUM_TO_DIAG[i]
        ss.append(f'{label}:{v:.3f}')
    return '\n'.join(ss)

class MyTrainer(Trainer):
    def prepare(self, **kwargs):
        self.font = ImageFont.truetype('/usr/share/fonts/ubuntu/UbuntuMono-R.ttf', 36)
        merge_weight = kwargs.pop('merge_weight', -1.0)
        if merge_weight > 0:
            self.criterion = NestedCrossEntropyLoss(
                reduction='mean',
                groups=[{
                    'weight': merge_weight,
                    'index': [[2, 3, 4]], # merge G,A,O label
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
        return CosineLRScheduler(
            self.optimizer,
            warmup_t=5, t_initial=max_epoch,
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
        parser.add_argument('--grid', '-g', type=int, default=768)
        parser.add_argument('--crop', '-c', type=int, default=768)
        parser.add_argument('--size', '-s', type=int, default=768)
        parser.add_argument('--dir', '-d', default='data/images')
        parser.add_argument('--merge-weight', '-w', type=float, default=-1.0)

    def create_loaders(self, num_classes):
        return self.as_loaders(*[
            LMGDataset(
                target=t,
                aug_mode=t,
                base_dir=self.a.dir,
                grid_size=self.a.grid,
                crop_size=self.a.crop,
                size=self.a.size,
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
            merge_weight=self.a.merge_weight,
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
        'epoch': 30,
        'lr': 0.001,
        'batch_size': 16,
        'save_period': 10,
    })
    cmd.run()
