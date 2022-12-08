import re
import torch
import timm
from torch import nn
from torchvision import transforms, models
from torch.nn import functional as F


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


class VGG(nn.Module):
    def __init__(self, name, num_classes=1, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        vgg = getattr(models, name)
        if not re.match(r'^vgg', name):
            raise Exception(f'At least starting with "vgg": {name}')
        if not vgg:
            raise Exception(f'Invalid model name: {name}')

        base = vgg(pretrained=pretrained)
        self.convs = base.features
        self.avgpool = base.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.num_classes),
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if self.num_classes > 1:
            x = torch.softmax(x, dim=1)
        else:
            x = torch.sigmoid(x)
        # x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        return x


def create_model(name, num_classes):
    if re.match(r'^vgg', name):
        return VGG(name=name, num_classes=num_classes)

    if m := re.match(r'^eff_(b[0-7])$', name):
        return EffNet(name=m[1], num_classes=num_classes)

    if m := re.match(r'^eff_(b[0-7]_ns)$', name):
        return EffNet(name=m[1], num_classes=num_classes)

    if m := re.match(r'^eff_(v2_b[0-4])$', name):
        return EffNet(name=m[1], num_classes=num_classes)

    if m := re.match(r'^eff_(v2_s|m|l)$', name):
        return EffNet(name=m[1], num_classes=num_classes)


available_models = \
    [f'eff_b{i}_ns' for i in range(8)] + \
    [f'eff_v2_b{i}' for i in range(4)] + \
    ['eff_v2_s', 'eff_v2_s','eff_v2_l' ] + \
    [f'vgg{i}' for i in [11, 13, 16, 19]] + \
    [f'vgg{i}_bn' for i in [11, 13, 16, 19]]

class CrossEntropyLoss(nn.Module):
    def __init__(self, eps=1e-24):
        super().__init__()
        self.eps = eps
        self.loss_fn = nn.NLLLoss()

    # y: target index
    def forward(self, x, y):
        return self.loss_fn((x + self.eps).log(), y)

def sumup_by_index(t, idx, rescale=False):
    # TODO: NEED EXPERIMENT!!!
    if rescale:
        group_sum = torch.mean(t[idx])
    else:
        group_sum = torch.sum(t[idx])
    new_idx = torch.ones(t.numel(), dtype=torch.bool)
    new_idx[idx] = False
    new_t = t[new_idx]
    if rescale:
        scale = t.numel() / (t.numel() - len(idx) + 1)
        new_t *= scale
    # append to last
    return torch.cat([new_t, torch.tensor([group_sum])])


class NestedCrossEntropyLoss(nn.Module):
    def __init__(self, groups=None, rescale=False, eps=1e-24, ):
        super().__init__()
        self.groups = groups
        self.rescale = rescale
        self.eps = eps

    def calc_cross_entropy(self, x, y):
        return - torch.sum(y * (x + self.eps).log())

    # y: one hot
    def forward(self, x, y):
        base = self.calc_cross_entropy(x, y)
        if not self.groups:
            return base

        ll = [base]
        for group in self.groups:
            new_x = sumup_by_index(x, group['index'], rescale=self.rescale)
            new_y = sumup_by_index(y, group['index'], rescale=self.rescale)
            group_loss = self.calc_cross_entropy(x, y)
            ll.append(group_loss * group['weight'])
        print(ll)
        return torch.tensor(ll).sum()


def compare_nested_vs_default():
    n = NestedCrossEntropyLoss(groups=[
        {
            'weight': 0.4,
            'index': [2, 3, 4],
        }
    ])

    x = F.softmax(torch.tensor([0.3, 0, 0, 0, 0.7]), dim=0)
    y = torch.tensor([1.0, 0, 0, 0, 0])

    print('n')
    print( n(x, y) )

    c = CrossEntropyLoss()
    print('c')
    print( c(x, torch.argmax(y)) )


def compare_nested():
    n = NestedCrossEntropyLoss(groups=[
        {
            'weight': 0.5,
            'index': [1, 2],
        }
    ])

    x0 = F.softmax(torch.tensor([0.0, 0.8, 0.1]), dim=0)
    y0 = torch.tensor([0.0, 1.0, 0.0])

    x1 = F.softmax(torch.tensor([0.1, 0.1, 0.8]), dim=0)
    y1 = torch.tensor([0.0, 1.0, 0.0])

    print('n x0 y0')
    print( n(x0, y0) )

    print('n x1 y1')
    print( n(x1, y1) )

    c = CrossEntropyLoss()
    print('c x0 y0')
    print( c(x0, torch.argmax(y0)) )

    print('c x1 y1')
    print( c(x1, torch.argmax(y1)) )
    exit(0)


if __name__ == '__main__':
    compare_nested()
    exit(0)

    model = create_model('eff_v2_b3', 3)
    count = sum(p.numel() for p in model.parameters()) / 1000000
    print(f'count: {count}M')
    x = torch.rand([2, 3, 512, 512])
    y = model(x)
    # loss = CrossEntropyLoss()
    # print('y', y, y.shape, 'loss', loss(y, torch.LongTensor([1, 1])))
    print('y', y, y.shape)
