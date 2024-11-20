import sys
import math
import re
from typing import NamedTuple, Callable

import torch
import timm
from timm.models.layers import to_2tuple, PatchEmbed
from torch import nn
from torch.nn import functional as F
import torchvision as tv
from torchvision import transforms, models
from pydantic import Field

from endaaman.ml import BaseMLCLI

from .ctranspath import ctranspath
from .loss import NestedCrossEntropyLoss, CrossEntropyLoss, SymmetricCosSimLoss, BarlowTwinsLoss


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

    def forward(self, x, activate=False, with_feature=False):
        features = self.base.forward_features(x)
        x = self.base.forward_head(features)

        if activate:
            if self.num_classes > 1:
                x = torch.softmax(x, dim=-1)
            else:
                x = torch.sigmoid(x)

        if with_features:
            pool = get_pool(self.base)
            features = pool(features)
            return x, features
        return x

class CompareModel(nn.Module):
    def __init__(self, num_classes, frozen=False, base='random'):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.Identity()
        self.frozen = frozen
        self.base_name = base
        if base == 'uni':
            # 'vit_large_patch16_224'
            self.base = timm.create_model('hf-hub:MahmoodLab/uni', pretrained=True, init_values=1e-5, dynamic_img_size=True)
        elif base == 'gigapath':
            # 'vit_giant_patch14_dinov2'
            self.base = timm.create_model('hf_hub:prov-gigapath/prov-gigapath', pretrained=True, dynamic_img_size=True)
        elif base == 'ctranspath':
            self.base = ctranspath(pretrained=True)
        elif base == 'baseline-cnn':
            self.base = timm.create_model('resnetrs50', pretrained=True)
            self.base.fc = nn.Identity()
        elif base == 'baseline-vit':
            # uni_kwargs = {
            #     'model_name': 'vit_large_patch16_224',
            #     'img_size': 224,
            #     'patch_size': 16,
            #     'init_values': 1e-5,
            #     'num_classes': 0,
            #     'dynamic_img_size': True,
            #     'pretrained': True,
            # }
            self.base = timm.create_model('vit_large_patch16_224', pretrained=True, dynamic_img_size=True)
            self.base.head = nn.Identity()
        elif base == 'random-vit':
            self.base = timm.create_model('vit_large_patch16_224', pretrained=False, dynamic_img_size=True)
            self.base.head = nn.Identity()
        elif base == 'random-cnn':
            self.base = timm.create_model('resnetrs50', pretrained=False)
            self.base.head = nn.Identity()
        elif base == 'baseline-swin':
            self.base = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
            self.base.head = nn.Identity()
            self.pool = nn.Sequential(
                tv.ops.Permute([0, 3, 1, 2]),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(start_dim=1),
            )
        else:
            self.base = timm.create_model(base, pretrained=True)
            self.base.fc = nn.Identity()

        if self.frozen:
            self.freeze_encoder()

        self.fc = nn.Linear(self.base.num_features, num_classes)

    def state_dict(self, *args, **kwargs):
        s = super().state_dict(*args, **kwargs)
        if not self.frozen:
            return s
        # if frozen, remove base weights
        return {k:v for k,v in s.items() if not k.startswith('base.')}

    def load_state_dict(self, state_dict, strict:bool = True, assign:bool = False):
        if not self.frozen:
            super().load_state_dict(state_dict, strict, assign)
        w = super().state_dict()
        w.update(state_dict)
        super().load_state_dict(w, strict, assign)

    def get_cam_layers(self):
        match self.base_name:
            case 'uni' | 'baseline-vit' | 'random-vit' | 'gigapath':
                return [self.base.blocks[-1].norm1]
            case 'baseline-cnn':
                return [self.base.layer4[-1]]
            # case _:
            #     raise RuntimeError('Invalid', _)
        return get_cam_layers(self.base, self.base_name)

    def freeze_encoder(self):
        self.frozen = True
        for param in self.base.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        self.frozen = False
        for param in self.base.parameters():
            param.requires_grad = True

    def forward(self, x, activate=False, with_features=False):
        features = self.base(x)
        features = self.pool(features)
        x = self.fc(features)

        if activate:
            if self.num_classes > 1:
                x = torch.softmax(x, dim=-1)
            else:
                x = torch.sigmoid(x)

        if with_features:
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

    def forward(self, x, activate=False, with_features=False):
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

        if with_features:
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
        for param in self.base.parameters():
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


class SimSiamModel(nn.Module):
    def __init__(self, name, num_features=2048, num_neck=512,  pretrained=True):
        super().__init__()
        self.name = name
        self.num_features = num_features
        self.num_neck = num_neck

        self.base = timm.create_model(name, pretrained=pretrained)
        num_prev = self.base.num_features
        self.pool = get_pool(self.base)

        self.projection_mlp = nn.Sequential(
            nn.Linear(num_prev, num_prev, bias=False),
            nn.BatchNorm1d(num_prev),
            nn.ReLU(inplace=True),
            nn.Linear(num_prev, num_prev, bias=False),
            nn.BatchNorm1d(num_prev),
            nn.ReLU(inplace=True),
            nn.Linear(num_prev, num_features, bias=False),
            nn.BatchNorm1d(num_features, affine=False)
        )

        self.prediction_mlp = nn.Sequential(
            nn.Linear(num_features, num_neck, bias=False),
            nn.BatchNorm1d(num_neck),
            nn.ReLU(inplace=True),
            nn.Linear(num_neck, num_features)
        )


    def forward_features(self, x, use_mlp=True):
        x = self.pool(self.base.forward_features(x))
        if use_mlp:
            x = self.projection_mlp(x)
            x = self.prediction_mlp(x)
        return x

    def forward(self, x01):
        x0 = x01[:, 0, ...] # B, P, C, H, W
        x1 = x01[:, 1, ...] # B, P, C, H, W

        f0 = self.pool(self.base.forward_features(x0))
        f1 = self.pool(self.base.forward_features(x1))

        z0 = self.projection_mlp(f0)
        z1 = self.projection_mlp(f1)

        p0 = self.prediction_mlp(z0)
        p1 = self.prediction_mlp(z1)
        return z0, z1, p0, p1


class BarlowTwinsModel(nn.Module):
    def __init__(self, name, num_features=2048, pretrained=True):
        super().__init__()
        self.name = name
        self.num_features = num_features

        self.base = timm.create_model(name, pretrained=pretrained)
        num_prev = self.base.num_features
        self.base.fc = nn.Identity()

        self.projection = nn.Sequential(
            nn.Linear(num_prev, num_features),
            nn.BatchNorm1d(num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features)
        )

    def forward_features(self, x, use_mlp=True):
        x = self.base(x)
        if use_mlp:
            x = self.projection(x)
        return x

    def forward(self, x, normalize=False):
        x = self.base(x)
        x = self.projection(x)
        if normalize:
            x = (x - x.mean(0)) / x.std(0)
        return x
