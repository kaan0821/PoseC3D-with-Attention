import torch
import torch.nn as nn
from torchsummary import summary


class Channel_attention(nn.Module):
    def __init__(self, channel, ratio=16):
        super(Channel_attention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.MLP = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, d, h, w = x.size()
        max_pool = self.max_pool(x).view([b, c])
        avg_pool = self.avg_pool(x).view([b, c])

        max_pool = self.MLP(max_pool)
        avg_pool = self.MLP(avg_pool)

        out = max_pool + avg_pool
        out = self.sigmoid(out).view([b, c, 1, 1, 1])
        return out * x


class Spacial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Spacial_attention, self).__init__()
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv3d(in_channels=2,
                              out_channels=1,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding,
                              bias=False
                              )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        out = torch.cat([max_pool, avg_pool], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return out * x


class CBAM_Block(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(CBAM_Block, self).__init__()
        self.channel_attention = Channel_attention(channel, ratio)
        self.spacial_attention = Spacial_attention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spacial_attention(x)
        return x


if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = CBAM_Block(channel=17).to(device)
    summary(net, input_size=(17, 32, 56, 56))

























