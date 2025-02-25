import os
import re

import numpy as np
import torch
from torch import nn


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
