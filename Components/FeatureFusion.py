from Components.ConvOps import ConvOps
from Components.DenseBlock import DenseBlock
from Components.InvBlock import InvBlock
from Components.SDB import SDB
from Components.DenseProjection_SGNet import DenseProjection_SGNet
import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    def __init__(self, channels, rgb_channels,scale):
        super(FeatureFusion, self).__init__()
        self.rgbprocess = nn.Conv2d(rgb_channels, rgb_channels, 3, 1, 1)
        self.rgbpre = nn.Conv2d(rgb_channels, rgb_channels, 1, 1, 0)
        self.spa_process = nn.Sequential(InvBlock(DenseBlock, channels + rgb_channels, channels),
                                         nn.Conv2d(channels + rgb_channels, channels, 1, 1, 0))
        self.fre_process = SDB(channels, rgb_channels)
        self.spa_att = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1, bias=True),
                                     nn.Sigmoid())
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = ConvOps.ChannelSTD
        self.cha_att = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=1, padding=0, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels, kernel_size=1, padding=0, bias=True),
                                     nn.Sigmoid())
        self.post = nn.Conv2d(channels, channels, 3, 1, 1)

        self.fuse_process = nn.Sequential(InvBlock(DenseBlock, 2*channels, channels),
                                         nn.Conv2d(2*channels, channels, 1, 1, 0))

        self.downBlock = DenseProjection_SGNet(channels, channels, scale, up=False, bottleneck=False)
        self.upBlock = DenseProjection_SGNet(channels, channels, scale, up=True, bottleneck=False)

    def forward(self, dp, rgb):

        dp = self.upBlock(dp)

        rgbpre = self.rgbprocess(rgb)
        rgb = self.rgbpre(rgbpre)
        spafuse = self.spa_process(torch.cat([dp, rgb], 1))
        frefuse = self.fre_process(dp, rgb)

        cat_f = torch.cat([spafuse, frefuse], 1)
        cat_f = self.fuse_process(cat_f)

        cha_res = self.cha_att(self.contrast(cat_f) + self.avgpool(cat_f)) * cat_f
        out = cha_res + dp

        out = self.downBlock(out)

        return out,rgbpre