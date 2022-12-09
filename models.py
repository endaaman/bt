import sys
import re
import torch
import timm
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, models


class EffNet(nn.Module):
    def __init__(self, name='v2_b0', num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        if m := re.match('^v2_(.+)$', name):
            model_name = f'tf_efficientnetv2_{m[1]}'
        else:
            model_name = f'tf_efficientnet_{name}'

        self.base = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    def get_cam_layer(self):
        return self.base.conv_head

    def forward(self, x):
        x =  self.base(x)
        if self.num_classes > 1:
            x = torch.softmax(x, dim=1)
        else:
            x = torch.sigmoid(x)
        return x


def create_model(name):
    m = re.match(r'^(.*)_(\d)$', name)
    if not m:
        raise ValueError(f'Invalid name: {name}')
    name = m[1]
    num_classes = int(m[2])

    if m := re.match(r'^eff_(b[0-7])$', name):
        return EffNet(name=m[1], num_classes=num_classes)

    if m := re.match(r'^eff_(b[0-7]_ns)$', name):
        return EffNet(name=m[1], num_classes=num_classes)

    if m := re.match(r'^eff_(v2_b[0-4])$', name):
        return EffNet(name=m[1], num_classes=num_classes)

    if m := re.match(r'^eff_(v2_s|m|l)$', name):
        return EffNet(name=m[1], num_classes=num_classes)

    raise ValueError(f'Invalid name: {name}')


available_models = \
    [f'eff_b{i}_ns' for i in range(8)] + \
    [f'eff_v2_b{i}' for i in range(4)] + \
    ['eff_v2_s', 'eff_v2_s','eff_v2_l' ]

class CrossEntropyLoss(nn.Module):
    def __init__(self, eps=1e-32):
        super().__init__()
        self.eps = eps

    # y: target index
    def forward(self, x, y):
        return F.nll_loss(torch.clamp(x, self.eps).log(), y)

def sumup_by_index(t, idxs, rescale=False):
    new_sums = []
    drop_idx = []
    for idx in idxs:
        new_sums.append(torch.sum(t[..., idx], dim=-1))
        drop_idx += idx

    new_idx = torch.ones(t.size(), dtype=torch.bool)
    new_idx[..., drop_idx] = False
    new_t = t[new_idx]

    # (1, 2, 3, 4, 5) [0, 1] -(sum)> (3, 3, 4, 5,)
    return torch.cat([new_t, torch.tensor(new_sums)])


class NestedCrossEntropyLoss(nn.Module):
    def __init__(self, groups=None, reduction='sum', rescale=False, by_index=True, eps=1e-32):
        super().__init__()
        self.groups = groups
        self.by_index = by_index
        self.rescale = rescale
        self.eps = eps

        self.reduce = {
            'mean': torch.mean,
            'sum': torch.sum,
        }[reduction]

    def calc_cross_entropy(self, x, y):
        return - torch.sum(y * torch.clamp(x, self.eps).log())

    def forward(self, x, y):
        if self.by_index:
            y = F.one_hot(y, x.size()[-1])
        base = self.calc_cross_entropy(x, y)
        if not self.groups:
            return base

        ll = [base]
        for group in self.groups:
            new_x = sumup_by_index(x, group['index'], rescale=self.rescale)
            new_y = sumup_by_index(y, group['index'], rescale=self.rescale)
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

    x0 = F.softmax(torch.tensor([0.8, 0.1, 0.1]), dim=-1)
    y0 = torch.tensor(2)
    x1 = F.softmax(torch.tensor([0.1, 0.8, 0.1]), dim=-1)
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
    compare_nested()
    sys.exit()

    model = create_model('eff_v2_b3_3')
    count = sum(p.numel() for p in model.parameters()) / 1000000
    print(f'count: {count}M')
    x_ = torch.rand([2, 3, 512, 512])
    y_ = model(x_)
    # loss = CrossEntropyLoss()
    # print('y', y, y.shape, 'loss', loss(y, torch.LongTensor([1, 1])))
    print('y', y_, y_.shape)
