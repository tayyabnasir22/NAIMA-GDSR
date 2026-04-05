from Components.DepthEncoder import DepthEncoder
from Components.RGBEncoder import RGBEncoder
from Components.SpatialProjector import SpatialProjectionAttention
import torch.nn as nn

class GTA(nn.Module):
    def __init__(
        self, num_feats: int, channel_scale: int, kernel_size: int):
        super(GTA, self).__init__()        
        self.spatial_projector = SpatialProjectionAttention(num_feats, channel_scale)
        self.depth_enc = DepthEncoder(channel_scale*num_feats, 6, kernel_size)
        self.rgb_encoder = RGBEncoder(num_feats, kernel_size)


    def forward(self, depth, token, rgb, H, W):
        dp_enc = self.depth_enc(depth)

        dp_spat = self.spatial_projector(
            token, H, W, dp_enc
        )

        rgb_enc = self.rgb_encoder(rgb)

        return dp_spat, rgb_enc