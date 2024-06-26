import torch.nn as nn
from arch.densenet201 import densenet201


class DenseNetRegression(nn.Module):
    def __init__(self, pretrained=True):
        super(DenseNetRegression, self).__init__()
        self.densenet = densenet201(pretrained=pretrained)
        self.densenet.classifier = nn.Linear(1920, 2)  # 修改最终输出

    def forward(self, x):
        return self.densenet(x)
