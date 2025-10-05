import torch
import torch.nn as nn
import torch.nn.functional as F


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample_LP(nn.Module):
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
移除 init_pos ：不再预先计算和存储初始位置偏移
，改为在 sample 方法中动态计算坐标。
动态计算坐标 ：在 sample 方法中，根据输入特征图的尺
寸（H, W）动态生成初始坐标 coords_h 和 coords_w，这样可以支持不同尺度的输入特征图。
保持坐标计算逻辑 ：将初始坐标与偏移量结合后进行归一化处
理，确保坐标在合理范围内，然后进行像素重排和网格采样。
'''
'''
1 文字阐述（逻辑链）
固定网格（保留 _init_pos）把亚像素采样点锁死在规则格点上 → 只能产生微小局部形变；
遥感场景中飞机、舰船、油罐往往存在大角度旋转、透视畸变 → 规则格点与目标轮廓错位；
删除 _init_pos 后，网络直接把像素中心当“锚点”，
可以自由地把采样点拉到任意位置 → 轮廓贴合度↑、小目标召回↑；
通过可视化采样点坐标（红色散点）可直观看到：
• 保留时 → 网格呈棋盘格；
• 删除后 → 散点沿目标长轴/边缘分布。
'''

'''
1. 动态偏移量的优势
原版的 init_pos 提供了一个固定的偏移量矩阵，这在某些情况下可能限制了模型对不同特征模式的适应能力。
改进版的 offset 去掉了固定的初始化偏移量，直接使用卷积层的输出作为偏移量。
这种动态偏移量可以根据输入特征图的不同进行自适应调整，从而更好地捕捉遥感图像中目标的多样性和复杂性。
2. 遥感图像的特点
遥感图像通常具有高分辨率和丰富的细节，
目标物体可能出现在不同的位置、尺度和方向上。
动态偏移量能够帮助模型更灵活地调整采样位置，从而更准确地检测和定位目标。
'''