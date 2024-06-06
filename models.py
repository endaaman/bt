import sys
import math
import re
from typing import NamedTuple, Callable

import torch
import timm
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, models
from pydantic import Field

from endaaman.ml import BaseMLCLI


def get_cam_layers(m, name=None):
    name = name or m.default_cfg['architecture']
    if re.match(r'.*efficientnet.*', name):
        return [m.conv_head]
    if re.match(r'^resnetrs.*', name):
        return [m.layer4[-1]]
    if re.match(r'^resnetv2.*', name):
        return [m.stages[-1].blocks[-1]]
    if re.match(r'^resnet\d+', name):
        return [m.layer4[-1].act2]
    if re.match(r'^caformer_.*', name):
        return [m.stages[-1].blocks[-1].res_scale2]
    if re.match(r'^convnext.*', name):
        # return [m.stages[-1].blocks[-1].conv_dw]
        return [m.stages[-1].blocks[-1].norm]
    if name == 'swin_base_patch4_window7_224':
        return [m.layers[-1].blocks[-1].norm1]
    raise RuntimeError('CAM layers are not determined.')
    return []

def get_pool(m):
    if hasattr(m, 'global_pool'):
        return m.global_pool
    return m.head.global_pool

class TimmModel(nn.Module):
    def __init__(self, name, num_classes, pretrained=True):
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        self.base = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        self.pool = get_pool(self.base)

    def get_cam_layers(self):
        return get_cam_layers(self.base, self.name)

    def forward(self, x, activate=False, with_feautres=False):
        features = self.base.forward_features(x)
        x = self.base.forward_head(features)

        if activate:
            if self.num_classes > 1:
                x = torch.softmax(x, dim=-1)
            else:
                x = torch.sigmoid(x)

        if with_feautres:
            pool = get_pool(self.base)
            features = pool(features)
            return x, features
        return x


class IICModel(nn.Module):
    def __init__(self, name, num_classes, num_classes_over, pretrained=True):
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        self.num_classes_over = num_classes_over
        self.base = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        if hasattr(self.base, 'global_pool'):
            self.pool = self.base.global_pool
        else:
            self.pool = self.base.head.global_pool

        self.num_features = self.base.num_features

        self.fc = nn.Linear(self.num_features, self.num_classes)
        self.fc_over = nn.Linear(self.num_features, self.num_classes_over)

    def get_cam_layers(self):
        return get_cam_layers(self.base, self.name)

    def forward(self, x, activate=False, with_feautres=False):
        features = self.base.forward_features(x)
        features = self.pool(features).flatten(1)

        x = self.fc(features)
        x_over = self.fc_over(features)

        if activate:
            if self.num_classes > 1:
                x = torch.softmax(x, dim=1)
                x_over = torch.softmax(x_over, dim=1)
            else:
                x = torch.sigmoid(x)
                x_over = torch.sigmoid(x_over)

        if with_feautres:
            return x, x_over, features
        return x, x_over


class MILModel(nn.Module):
    def __init__(self, name, num_classes, activation='softmax', params_count=512, pretrained=True):
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        self.activation = activation
        self.params_count = params_count

        self.base = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        self.num_features = self.base.num_features
        self.classifier = nn.Linear(self.num_features, self.num_classes)

        self.u = nn.Linear(self.num_features, self.params_count, bias=False)
        self.v = nn.Linear(self.num_features, self.params_count, bias=False)
        self.w = nn.Linear(self.params_count, 1, bias=False)

    def set_freeze_base_model_weights(self, flag):
        for param in base.parameters():
            param.requires_grad = flag

    def get_cam_layers(self):
        return get_cam_layers(self.base, self.name)

    def compute_attentions(self, features):
        # raise NotImplementedError('DO IMPLEMENT')
        aa = []
        for feature in features:
            xu = self.u(feature)
            xv = torch.sigmoid(self.v(feature))
            alpha = self.w(xu * xv)
            aa.append(alpha)
        aa = torch.stack(aa).flatten()
        return aa

    def compute_attentions_and_activate(self, features):
        aa = self.compute_attentions(features)

        if self.activation == 'softmax':
            aa = torch.softmax(aa, dim=-1)
        elif self.activation == 'sigmoid':
            aa = torch.sigmoid(aa)
        elif self.activation == 'raw':
            pass
        else:
            raise RuntimeError('Invalid activation:', self.activation)
        return aa

    def forward(self, x, activate=False):
        features = self.base.forward_features(x)

        if hasattr(self.base, 'global_pool'):
            pool = self.base.global_pool
        else:
            pool = self.base.head.global_pool

        features = pool(features)

        aa = self.compute_attentions_and_activate(features)

        feature = (features * aa[:, None]).sum(dim=0)
        y = self.classifier(feature)
        yy = self.classifier(features)

        if activate:
            if self.num_classes > 1:
                y = torch.softmax(y, dim=-1)
                yy = torch.softmax(yy, dim=-1)
            else:
                y = torch.sigmoid(y)
                yy = torch.sigmoid(yy)

        return {
            'y': y,
            'yy': yy,
            'feature': feature,
            'features': features,
            'attentions': aa,
        }

class GraphMatrix(nn.Module):
    def __init__(self, size, initial_value=None):
        super().__init__()
        self.size = size
        self.gamma = 2
        # self.depth = depth
        self.mask = nn.Parameter(torch.triu(torch.ones(size, size), diagonal=0))
        self.mask.requires_grad = False
        if initial_value is None:
            param = torch.normal(mean=1, std=math.sqrt(2/size/size), size=(size, size))
        else:
            param = initial_value
        param[self.mask<1] = 0.0
        self.matrix = nn.Parameter(param)

    def get_matrix(self):
        m = self.matrix * self.mask
        m =(m+m.t())/2
        # return m.clamp_(1e-16)
        return m

    def calc_loss(self, preds, gts, m):
        preds = torch.matmul(preds, m)
        gts = torch.matmul(gts, m)
        loss = preds * (preds.log() - gts.log())
        # focal
        loss = (1 - preds * gts) ** self.gamma
        loss = loss.sum(dim=-1)
        return loss.mean()

    def forward(self, preds, gts, by_index=True):
        if by_index:
            gts = F.one_hot(gts, num_classes=preds.shape[-1]).float()
        m = self.get_matrix()

        # dim0
        return self.calc_loss(preds, gts, m.softmax(dim=0))

        # dim1
        # return self.calc_loss(preds, gts, m.softmax(dim=1))

        # sigmoid
        # return self.calc_loss(preds, gts, torch.sigmoid(m))

        # dual
        # loss0 = self.calc_loss(preds, gts, m.softmax(dim=0))
        # loss1 = self.calc_loss(preds, gts, m.softmax(dim=1))
        # return (loss0 + loss1) / 2


class TimmModelWithGraph(nn.Module):
    def __init__(self, model, graph_matrix):
        super().__init__()
        self.model = model
        self.graph_matrix = graph_matrix

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)



class TimmModelWithHier(nn.Module):
    def __init__(self, model, graph_matrix):
        super().__init__()
        self.model = model
        self.hier_matrixes = hier_matrixes

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


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
        # logits=[1, 2, 3, 4, 5] -> [(1 + 2), 3, (4 + 5)] -> [3, 3, 9]
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


class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
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
        name : str = Field('tf_efficientnetv2_b0', s='-n')

    def run_model(self, a):
        # model = AttentionModel(name=a.model, num_classes=3)
        # y, aa = model(x, with_attentions=True)

        model = IICModel(name=a.name, num_classes=3, num_classes_over=10)
        x = torch.rand([4, 3, 512, 512])
        y, y_over, f = model(x, with_feautres=True)
        print('y', y.shape)
        print('y_over', y_over.shape)
        print('f', f.shape)

    def run_mat(self, a):
        initial_value = torch.tensor([
            [10, .0, .0, .0, .0, ],
            [.0, 10, .0, .0, .0, ],
            [.0, .0, 5, 10, .0, ],
            [.0, .0, .0, 5, .0, ],
            [.0, .0, .0, .0, 10, ],
        ]).float().clamp(1e-16)
        g = GraphMatrix(5, initial_value)
        # preds = torch.randn([3, 5]).softmax(dim=1)
        preds = torch.tensor([
            [.6, .1, .1, .1, .1],
            [.1, .6, .1, .1, .1],
            [.1, .1, .6, .1, .1],
            [.1, .1, .1, .6, .1],
            [.1, .1, .1, .1, .6],
        ])
        gts = torch.tensor([2, 2, 2, 2, 2])
        print(g(preds, gts, by_index=True))


if __name__ == '__main__':
    cli = CLI()
    cli.run()
