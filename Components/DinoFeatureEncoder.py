import torch.nn as nn

class DinoFeatureEncoder(nn.Module):
    def __init__(self, embeding_dim: int = 384, num_feats: int = 48, scale: int = 4, patch_size: int = 14):
        super(DinoFeatureEncoder, self).__init__()
        
        self.patch_size = patch_size

        self.proj1 = nn.Conv2d(embeding_dim, embeding_dim*2, kernel_size=3)  # r=2
        
        self.proj2 = nn.Conv2d(embeding_dim*2, num_feats * (scale ** 2), kernel_size=1)

        self.upsample = nn.PixelShuffle(scale)

    def forward(self, feature, h, w):
        B, N, C = feature.shape
        h = h // self.patch_size
        w = w // self.patch_size

        x = feature.permute(0, 2, 1).reshape(B, C, h, w)

        x = self.proj1(x)
        x = self.proj2(x)

        return self.upsample(x)