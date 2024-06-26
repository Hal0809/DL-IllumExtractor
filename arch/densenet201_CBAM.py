import torch
import torch.nn as nn
from torch import Tensor
from arch.densenet201 import densenet201
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, num_input_features):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(num_input_features, num_input_features // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(num_input_features // 16, num_input_features, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class DenseNet_CBAM(nn.Module):
    def __init__(self, pretrained=True):
        super(DenseNet_CBAM, self).__init__()
        self.densenet = densenet201(pretrained=pretrained)
        self.ca = ChannelAttention(1920)
        self.sa = SpatialAttention()
        self.densenet.add_module("ca", self.ca)
        self.densenet.add_module("sa", self.sa)
        self.densenet.classifier = nn.Linear(1920, 2)  # 修改最终输出

    def forward(self, features):
        features = self.densenet.features(features)
        features = self.ca(features) * features
        features = self.sa(features) * features
        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        out = self.densenet.classifier(out)
        return out
