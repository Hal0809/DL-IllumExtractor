import torch
from torch import nn
from arch.densenet201 import densenet201


class RVFL(nn.Module):
    def __init__(self, input_dim, enhance_dim, output_dim):
        super(RVFL, self).__init__()
        self.enhance_nodes = nn.Linear(input_dim, enhance_dim, bias=True)
        self.output = nn.Linear(input_dim+enhance_dim, output_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        enhance_nodes = self.enhance_nodes(x)
        inputs = torch.cat([x, enhance_nodes], dim=-1)
        output = self.output(inputs)
        output = self.relu(output)
        return output


class DenseNet_RVFL(nn.Module):
    def __init__(self, pretrained=True):
        super(DenseNet_RVFL, self).__init__()
        self.densenet = densenet201(pretrained=pretrained)
        self.densenet.classifier = nn.Linear(1920, 1920)
        self.rvfl = RVFL(1920, 1920, 2)

    def forward(self, x):
        x = self.densenet(x)
        x = self.rvfl(x)
        return x
