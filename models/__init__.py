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
from .compare import CompareModel
from .vit import ViT, create_vit



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
