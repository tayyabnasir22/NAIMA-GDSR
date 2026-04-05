from Components.DinoFeatureEncoder import DinoFeatureEncoder
from Components.TRGRAMKQV import TRGRAMKQV
import torch.nn as nn
import torch.nn.functional as F

class SpatialProjectionAttention(nn.Module):
    def __init__(
        self, num_feats: int, channel_scale: int):

        super(SpatialProjectionAttention, self).__init__()
        self.token_encoder = DinoFeatureEncoder()
        self.att = TRGRAMKQV(channel_scale * num_feats, num_feats)

    def forward(self, token, H, W, depth_feat):
        tk_encoded = self.token_encoder(token, H, W)

        tk_encoded = F.interpolate(
                    tk_encoded,
                    size=(depth_feat.shape[-2], depth_feat.shape[-1]),
                    mode="bilinear",
                    align_corners=False
                )

        return self.att(depth_feat, tk_encoded)