import sys
import re
from typing import NamedTuple, Callable

import torch
import timm
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, models


class ModelId(NamedTuple):
    name: str
    num_classes: int

    def __str__(self):
        return f'{self.name}_{self.num_classes}'

    @classmethod
    def from_str(cls, s):
        m = re.match(r'^(.*)_(\d+)$', s)
        if not m:
            raise ValueError(f'Invalid model id: {s}')
        return ModelId(m[1], int(m[2]))

class TimmModel(nn.Module):
    def __init__(self, name='tf_efficientnetv2_b0', num_classes=1, activation=True):
        super().__init__()
        self.num_classes = num_classes
        self.activation = activation
        self.base = timm.create_model(name, pretrained=True, num_classes=num_classes)

    def get_cam_layer(self):
        return self.base.conv_head

    def forward(self, x, activate=True):
        x = self.base(x)
        if activate:
            if self.num_classes > 1:
                x = torch.softmax(x, dim=1)
            else:
                x = torch.sigmoid(x)
        return x


def create_model(model_id):
    return TimmModel(name=model_id.name, num_classes=model_id.num_classes)


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
    def __init__(self, groups=None, reduction='sum', sum_logits=True, by_index=True, eps=1e-32):
        super().__init__()
        self.groups = groups
        self.by_index = by_index
        self.sum_logits = sum_logits
        self.eps = eps
        self.C = 1

        self.reduce = {
            'mean': torch.mean,
            'sum': torch.sum,
        }[reduction]

    def sumup_by_index(self, t, idxs):
        new_sums = []
        drop_idx = []
        for idx in idxs:
            target = t[..., idx]
            new_sums.append(torch.sum(target, dim=-1))
            drop_idx += idx

        new_idx = torch.ones(t.size(), dtype=torch.bool)
        new_idx[..., drop_idx] = False
        rest_t = t[new_idx]

        # (1, 2, 3, 4, 5) [0, 1] -(sum)> (3, 3, 4, 5,)
        new_t = torch.cat([rest_t, torch.tensor(new_sums)])

        return new_t

    def calc_cross_entropy(self, x, y):
        x = torch.softmax(x, dim=-1)
        return - torch.sum(y * torch.clamp(x, self.eps).log())

    def forward(self, x, y):
        """
        x: logits
        y: {yc|0<yc<1}
        """
        if self.by_index:
            y = F.one_hot(y, x.size()[-1])
        base = self.calc_cross_entropy(x, y)
        if not self.groups:
            return base

        ll = [base]
        for group in self.groups:
            new_x = self.sumup_by_index(x, group['index'])
            new_y = self.sumup_by_index(y, group['index'])
            group_loss = self.calc_cross_entropy(new_x, new_y)
            ll.append(group_loss * group['weight'])
        return self.reduce(torch.tensor(ll))


def compare_nested():
    n = NestedCrossEntropyLoss(
        reduction='sum',
        groups=[
            {
                'weight': 1.0,
                'index': [[1, 2]],
            }
        ])

    x0 = torch.tensor([2.0, 1, 1])
    y0 = torch.tensor(2)
    x1 = torch.tensor([1, 2.0, 1])
    y1 = torch.tensor(2)

    print('n x0 y0')
    print( n(x0, y0) )
    print('n x1 y1')
    print( n(x1, y1) )

    c = CrossEntropyLoss()
    print('c x0 y0')
    print( c(x0, y0) )
    print('c x1 y1')
    print( c(x1, y1) )


if __name__ == '__main__':
    # compare_nested()
    model_id = ModelId('mo', 1)
    print(str(model_id))
    sys.exit()

    model = create_model('eff_v2_b3_3')
    count = sum(p.numel() for p in model.parameters()) / 1000000
    print(f'count: {count}M')
    x_ = torch.rand([2, 3, 512, 512])
    y_ = model(x_)
    # loss = CrossEntropyLoss()
    # print('y', y, y.shape, 'loss', loss(y, torch.LongTensor([1, 1])))
    print('y', y_, y_.shape)
