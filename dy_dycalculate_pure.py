import torch
import torch.nn as nn
import torch.nn.functional as F


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, groups=4):
        super().__init__()
        self.scale = scale
        self.groups = groups
        assert in_channels >= groups and in_channels % groups == 0

        out_channels = 2 * groups * scale ** 2
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H, device=x.device) + 0.5
        coords_w = torch.arange(W, device=x.device) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward(self, x):
        offset = self.offset(x) * 0.25
        return self.sample(x, offset)

'''
Remove init_pos: No longer pre-calculate and store the initial position offset

Change to dynamically calculate the coordinates in the sample method.

Dynamic calculation of coordinates: In the sample method, the initial coordinates coords_h 
and coords_w are dynamically generated based on the size (H, W) of the input feature map, which can support input feature maps of different scales.

Maintain the coordinate calculation logic: After combining the initial coordinates with the offset,
perform normalization processing to ensure that the coordinates are within a reasonable range, and then carry out pixel rearrangement and grid sampling.
'''

'''
1. The advantages of dynamic offset
The original init_pos provides a fixed offset matrix, which may limit the model's adaptability to different feature patterns in some cases.
The improved offset eliminates the fixed initialization offset and directly uses the output of the convolutional layer as the offset.
This dynamic offset can be adaptively adjusted according to different input feature maps, thereby better capturing the diversity and complexity of targets in remote sensing images.
2. Characteristics of remote sensing Images
Remote sensing images usually have high resolution and rich details.
The target object may appear at different positions, scales and directions.
Dynamic offsets can help the model adjust the sampling position more flexibly, thereby detecting and locating the target more accurately.

'''
