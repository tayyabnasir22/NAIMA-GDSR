from Components.DepthEncoder import DepthEncoder
from Components.ConvOps import ConvOps
import torch.nn as nn

class UpsamplerBlock(nn.Module):
    def __init__(
        self, num_feats: int, kernel_size: int, scale):
        super(UpsamplerBlock, self).__init__()        
        inp_channels = 3*num_feats
        out_channels = 2*num_feats
        self.upsampler1 = ConvOps.ProjectionConv(inp_channels, out_channels, scale, True)
        self.act1 = nn.PReLU(out_channels)
        self.conv1 = ConvOps.ProjectionConv(out_channels, inp_channels, scale, False)
        self.act2 = nn.PReLU(inp_channels)
        self.upsampler2 = ConvOps.ProjectionConv(inp_channels, out_channels, scale, True)
        self.act3 = nn.PReLU(out_channels)


        self.enc1 = DepthEncoder(2*num_feats, 8, kernel_size)
        self.enc2 = DepthEncoder(2*num_feats, 8, kernel_size)
        self.enc3 = DepthEncoder(2*num_feats, 8, kernel_size)
        
        

        self.conv2 = ConvOps.DefaultConv(2*num_feats, num_feats, kernel_size, bias=True)
        self.act4 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv3 = ConvOps.DefaultConv(num_feats, 1, kernel_size, bias=True)


    def forward(self, x):
        # Upsample
        u1 = self.upsampler1(x)
        u1 = self.act1(u1)

        u2 = self.conv1(u1)
        u2 = self.act2(u2)
        u2 = u2 - x

        u3 = self.upsampler2(u2)
        u3 = self.act3(u3)

        x = u3 + u1

        # Conv proj
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        x = self.conv2(x)
        x = self.act4(x)

        return self.conv3(x)