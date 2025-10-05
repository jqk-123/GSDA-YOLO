import math
import torch
import torch.nn as nn

def autopad(k, p=None, d=1):
    """
    Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.
    `k`: kernel, `p`: padding, `d`: dilation.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initializes a standard convolution layer with optional batch normalization and activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Applies a fused convolution and activation function to the input tensor `x`."""
        return self.act(self.conv(x))

class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
class GhostConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g=g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, g=c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GSConv_GhostConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = GhostConv(c1, c_, k, s, g=g, act=act)
        self.cv2 = GhostConv(c_, c_, 5, 1, g=g, act=act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        x2 = torch.cat((x1, x2), 1)
        # shuffle
        b, n, h, w = x2.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
        return torch.cat((y[0], y[1]), 1)


class GSConvns_GhostConv(GSConv_GhostConv):
    # GSConv with a normative-shuffle https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__(c1, c2, k, s, p, g, act=True)
        c_ = c2 // 2
        self.shuf = nn.Conv2d(c_ * 2, c2, 1, 1, 0, bias=False)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # normative-shuffle, TRT supported
        return nn.ReLU()(self.shuf(x2))


class GSBottleneck_GhostConv(nn.Module):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConv_GhostConv(c1, c_, 1, 1),
            GSConv_GhostConv(c_, c2, 3, 1, act=False))
        self.shortcut = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)


class GSBottleneckns_GhostConv(GSBottleneck_GhostConv):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__(c1, c2, k, s, e)
        c_ = int(c2 * e)
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConvns_GhostConv(c1, c_, 1, 1),
            GSConvns_GhostConv(c_, c2, 3, 1, act=False))


class GSBottleneckC_GhostConv(GSBottleneck_GhostConv):
    # cheap GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__(c1, c2, k, s)
        self.shortcut = DWConv(c1, c2, k, s, act=False)


class VoVGSCSP_GhostConv(nn.Module):
    # VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck_GhostConv(c_, c_, e=1.0) for _ in range(n)))
        self.res = Conv(c_, c_, 3, 1, act=False)
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))


class VoVGSCSPns_GhostConv(VoVGSCSP_GhostConv):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.gsb = nn.Sequential(*(GSBottleneckns_GhostConv(c_, c_, e=1.0) for _ in range(n)))


class VoVGSCSPC_GhostConv(VoVGSCSP_GhostConv):
    # cheap VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2)
        c_ = int(c2 * 0.5)  # hidden channels
        self.gsb = GSBottleneckC_GhostConv(c_, c_, 1, 1)


'''
新增 GhostConv 模块：
GhostConv 使用两个卷积层来生成更多的特征图，通过减少参数数量来提高计算效率。
cv1 是一个标准卷积，用于生成基础特征图。
cv2 是一个深度可分离卷积，用于生成更多的特征图。
修改 GSConv 模块为 GSConvWithGhostConv：
将原有的 Conv 模块替换为 GhostConv，以减少参数数量并提高计算效率。
通过 GhostConv 生成的特征图进行拼接和通道重排，保持了 GSConv 的原始结构和功能。
通过引入 GhostConv，GSConvWithGhostConv 模块在保持较高特征提取能力的同时，
减少了参数数量和计算量，适合用于轻量级的实时目标检测任务。
'''