import torch
import torch.nn as nn

class FreDiff(nn.Module):
    def __init__(self, channels,rgb_channels):
        super(FreDiff, self).__init__()

        self.fuse_c = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.fuse_sub = nn.Sequential(nn.Conv2d(channels,channels,1,1,0),nn.LeakyReLU(0.1,inplace=False),
                                      nn.Conv2d(channels,channels,1,1,0))
        self.post = nn.Conv2d(2*channels,channels,1,1,0)
        self.pre_rgb = nn.Conv2d(rgb_channels,channels,1,1,0)
        self.pre_dep = nn.Conv2d(channels,channels,1,1,0)

        self.sig = nn.Sigmoid()

    def forward(self, dp, rgb):

        dp1 = self.pre_dep(dp)
        rgb1 = self.pre_rgb(rgb)

        fuse_c = self.fuse_c(dp1)

        fuse_sub = self.fuse_sub(torch.abs(rgb1 - dp1))
        cat_fuse = torch.cat([fuse_c,fuse_sub],1)

        return self.post(cat_fuse)