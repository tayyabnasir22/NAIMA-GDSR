from Components.FreDiff import FreDiff
import torch
import torch.nn as nn

# From SGNet
class SDB(nn.Module):
    def __init__(self, channels,rgb_channels):
        super(SDB, self).__init__()
        self.pre1 = nn.Conv2d(channels,channels,1,1,0)
        self.pre2 = nn.Conv2d(rgb_channels,rgb_channels,1,1,0)
        self.amp_fuse = FreDiff(channels,rgb_channels)
        self.pha_fuse = FreDiff(channels,rgb_channels)
        self.post = nn.Conv2d(channels,channels,1,1,0)

    def forward(self, dp, rgb):

        _, _, H, W = dp.shape
        dp = torch.fft.rfft2(self.pre1(dp)+1e-8, norm='backward')
        rgb = torch.fft.rfft2(self.pre2(rgb)+1e-8, norm='backward')
        dp_amp = torch.abs(dp)
        dp_pha = torch.angle(dp)
        rgb_amp = torch.abs(rgb)
        rgb_pha = torch.angle(rgb)
        amp_fuse = self.amp_fuse(dp_amp,rgb_amp)
        pha_fuse = self.pha_fuse(dp_pha,rgb_pha)

        real = amp_fuse * torch.cos(pha_fuse)+1e-8
        imag = amp_fuse * torch.sin(pha_fuse)+1e-8
        out = torch.complex(real, imag)+1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))

        return self.post(out)