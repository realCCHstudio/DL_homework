import torch
import torch.nn as nn

# DFPN 模块（密集特征金字塔网络）
class DFPN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DFPN, self).__init__()
        self.downsample = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.upsample = nn.ConvTranspose2d(out_channels, in_channels, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x_down = self.downsample(x)
        x_up = self.upsample(x_down)
        return x + x_up

# DAL 模块（密集注意力层），动态输入通道数
class DAL(nn.Module):
    def __init__(self, in_channels):
        super(DAL, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 全连接层动态调整通道数
        self.fc = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)
