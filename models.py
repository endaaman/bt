import sys
import re
from typing import NamedTuple, Callable

import torch
import timm
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, models
from endaaman.ml import BaseCLI


class TimmModel(nn.Module):
    def __init__(self, name, num_classes, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.base = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

    def get_cam_layer(self):
        return self.base.conv_head

    def forward(self, x, activate=False):
        x = self.base(x)
        if activate:
            if self.num_classes > 1:
                x = torch.softmax(x, dim=1)
            else:
                x = torch.sigmoid(x)
        return x


class AttentionModel(nn.Module):
    def __init__(self, name, num_classes, params_count=10, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.base = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        self.num_features = self.base.num_features
        self.params_count = params_count

        self.u = nn.Parameter(torch.randn(params_count, self.num_features))
        self.v = nn.Parameter(torch.randn(params_count, self.num_features))
        self.w = nn.Parameter(torch.randn(params_count, 1))
        self.fc = nn.Linear(self.num_features, self.num_classes)

    def get_cam_layer(self):
        return self.base.conv_head

    def compute_attention_scores(self, x):
        xu = torch.tanh(torch.matmul(self.u, x))
        xv = torch.sigmoid(torch.matmul(self.v, x))
        x = xu * xv
        alpha = torch.matmul(x, self.w)
        return alpha

    def compute_attentions(self, features):
        aa = []
        for feature in features:
            aa.append(self.compute_attention_scores(feature))
        aa = torch.stack(aa).flatten()
        aa = torch.softmax(aa, dim=0)
        return aa

    def forward(self, x, activate=False, with_attentions=False):
        x = self.base.forward_features(x)
        x = self.base.forward_head(x, pre_logits=True)
        aa = self.compute_attentions(x)
        x = x * aa[:, None]
        x = x.sum(dim=0)
        x = self.fc(x)
        if activate:
            if self.num_classes > 1:
                x = torch.softmax(x, dim=1)
            else:
                x = torch.sigmoid(x)
        if with_attentions:
            return x, aa
        return x



class CrossEntropyLoss(nn.Module):
    def __init__(self, eps=1e-32, input_logits=True):
        super().__init__()
        self.eps = eps
        self.input_logits = input_logits

    # y: target index
    def forward(self, x, y):
        if self.input_logits:
            x = torch.softmax(x, dim=-1)
        return F.nll_loss(torch.clamp(x, self.eps).log(), y)



class NestedCrossEntropyLoss(nn.Module):
    def __init__(self, rules=None, sum_logits=True, by_index=True, eps=1e-32):
        super().__init__()
        # Ex. rules: [{'weight': 1.0, 'index': [[0, 1], [3, 4]]}]
        #     This means to merge 0 and 1, 3 and 4
        # The logits should be like below
        # [1, 2, 3, 4, 5] -> [(1 + 2), 3, (4 + 5)] -> [3, 3, 9]
        self.rules = rules
        self.by_index = by_index
        self.sum_logits = sum_logits
        self.eps = eps
        self.C = 1

    def sumup_by_index(self, logits, idxs):
        # Ex: logits (1, 2, 3, 4, 5, ) index [2, 3, 4]
        merged_logits = []
        drop_idx = []

        # Sum up logits specified by param index

        for idx in idxs:
            merging_logits = logits[..., idx]
            merged = torch.sum(merging_logits, dim=-1)[..., None]
            merged_logits.append(merged)
            drop_idx += idx
        # Ex: merged_logits ((3+4+5), )

        new_idx = torch.ones(logits.size(), dtype=torch.bool)
        new_idx[..., drop_idx] = False
        rest_logits = logits[new_idx]
        if len(logits.shape) > 1:
            # Need to reshape because rest_logits is flattened
            rest_logits = rest_logits.view(*logits.shape[:-1], -1)
        # Ex: rest_logits (1, 2, )

        logits = torch.cat([rest_logits, *merged_logits], dim=-1)
        # Ex: (1, 2, (3+4+5)) -> (1, 2, 12)
        return logits

    def calc_cross_entropy(self, x, y):
        x = torch.softmax(x, dim=-1)
        return - torch.sum(y * torch.clamp(x, self.eps).log()) / len(x)

    def forward(self, x, y):
        """
        x: logits
        y: {yc|0<yc<1}
        """
        if self.by_index:
            y = F.one_hot(y, x.size()[-1])
        base = self.calc_cross_entropy(x, y)
        if not self.rules:
            return base

        ll = []
        total_weight = 0.0
        for rule in self.rules:
            new_x = self.sumup_by_index(x, rule['index'])
            new_y = self.sumup_by_index(y, rule['index'])
            loss_by_rule = self.calc_cross_entropy(new_x, new_y)
            l = loss_by_rule * rule['weight']
            total_weight += rule['weight']
            ll.append(l)
        loss = torch.sum(torch.stack(ll))
        loss /= total_weight
        return loss


class CLI(BaseCLI):
    class CommonArgs(BaseCLI.CommonArgs):
        pass

    def run_loss(self, a:CommonArgs):
        n = NestedCrossEntropyLoss(
            rules=[
                {
                    'weight': 1000.0, 'index': [],
                },
                {
                    'weight': 1.0, 'index': [[2, 3, 4]],
                }
            ])

        # loss should be different
        x0 = torch.tensor([[1., 0, 0, 3, 0], [1., 0, 0, 3, 0]])
        y0 = torch.tensor([2, 2])

        # loss should be same
        x1 = torch.tensor([[1., 0, 0, 0, 3], [1., 0, 0, 0, 3]])
        y1 = torch.tensor([1, 1])

        print('nested')
        print('x0 y0')
        print( n(x0, y0) )
        print('x1 y1')
        print( n(x1, y1) )

        c = CrossEntropyLoss()
        print()
        print('normal')
        print('x0 y0')
        print( c(x0, y0) )
        print('x1 y1')
        print( c(x1, y1) )

    class ModelArgs(CommonArgs):
        model : str = 'tf_efficientnetv2_b0'

    def run_model(self, a):
        model = AttentionModel(name=a.model, num_classes=3)
        x = torch.rand([4, 3, 512, 512])
        y, aa = model(x, with_attentions=True)
        print('y', y, y.shape)
        print('aa', aa, aa.shape)


if __name__ == '__main__':
    cli = CLI()
    cli.run()
