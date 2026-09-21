"""Shared custom-layer registration for MicroCharNet checkpoint tooling."""

from __future__ import annotations

import torch
from torch import nn


class HSigmoid(nn.Module):
    """Compatibility implementation used by the trained checkpoint."""

    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3.0) / 6.0


class HSwish(nn.Module):
    """Compatibility implementation used by the trained checkpoint."""

    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.sigmoid = HSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """Coordinate Attention layer serialized by ``microcharnet.pt``."""

    def __init__(self, inp: int, oup: int, reduction: int = 32) -> None:
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = HSwish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        _, _, height, width = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [height, width], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        attention_h = self.conv_h(x_h).sigmoid()
        attention_w = self.conv_w(x_w).sigmoid()
        return identity * attention_w * attention_h


def register_custom_layers() -> None:
    """Register names required while loading/exporting the checkpoint."""

    import ultralytics.nn.modules.block as block

    block.CoordAtt = CoordAtt
    block.h_swish = HSwish
    block.h_sigmoid = HSigmoid
