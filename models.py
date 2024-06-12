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


class HierMatrixes(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        assert size > 2
        # self.matrixes = []
        self.matrixes = nn.ParameterList()
        for level in range(2, size):
            param = torch.normal(mean=1, std=math.sqrt(2/size/size), size=(size, level))
            m = nn.Parameter(param)
            m.requires_grad = True
            self.matrixes.append(m)

    def forward(self, preds, gts):
        losses = []
        m_losses = []
        gts = F.one_hot(gts, num_classes=preds.shape[-1]).float()
        for m in self.matrixes:
            m = torch.softmax(m, dim=1)
            pred = torch.matmul(preds, m).softmax(dim=1)
            gt = torch.matmul(gts, m)
            loss = -(pred.log() * gt).sum()
            losses.append(loss)
            m_losses.append(m.mean(dim=0).std())
            m_losses.append(-m.std(dim=0).mean())
        loss = torch.stack(losses).mean()
        m_loss = torch.stack(m_losses).sum()
        return loss + m_loss


class TimmModelWithHier(nn.Module):
    def __init__(self, model, hier_matrixes):
        super().__init__()
        self.model = model
        self.hier_matrixes = hier_matrixes

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


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

    def run_hier(self, a):
        model = TimmModel('resnet18', 6)
        hier_matrixes = HierMatrixes(6)
        h = TimmModelWithHier(model, hier_matrixes)

        ii = torch.randn(4, 3, 256, 256)
        ll = h(ii)

        gts = torch.tensor([0, 1, 2, 3])

        loss = hier_matrixes(ll, gts)
        print(ll.shape)
        print(loss)


if __name__ == '__main__':
    cli = CLI()
    cli.run()
