import torch
from torch import nn
import torch.nn.functional as F
from arch.densenet201 import densenet201
from arch.densenet201_CBAM import ChannelAttention, SpatialAttention
from arch.densenet201_RVFL import RVFL


class DenseNet_CBAM_RVFL(nn.Module):
    def __init__(self, pretrained=True):
        super(DenseNet_CBAM_RVFL, self).__init__()
        self.densenet = densenet201(pretrained=pretrained)
        self.ca = ChannelAttention(1920)
        self.sa = SpatialAttention()
        self.densenet.add_module("ca", self.ca)
        self.densenet.add_module("sa", self.sa)
        self.densenet.classifier = nn.Linear(1920, 1920)
        self.rvfl = RVFL(1920, 1920, 2)

    def forward(self, features):
        features = self.densenet.features(features)
        features = self.ca(features) * features
        features = self.sa(features) * features
        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        out = self.densenet.classifier(out)
        out = self.rvfl(out)
        return out
