from Components.GTA import GTA
from Components.UpsamplerBlock import UpsamplerBlock
import torch.nn as nn
import torch
from Components.ConvOps import ConvOps
from Components.DinoV2.DinoVisionTransformer import DinoVisionTransformer
from Components.FeatureFusion import FeatureFusion

class NAIMA(nn.Module):
    def __init__(
            self, 
            use_pretrained: bool, 
            num_feats: int = 48, 
            img_size: int = 280, 
            patch_size: int = 14, 
            scale: int = 4,
            kernel_size = 3,
            freeze_dino: bool = True
        ):
        super(NAIMA, self).__init__()
        self.patch_size = patch_size

        self.intermediate_layer_idx = [2, 5, 8, 11]
        
        self.semantics_encoder = DinoVisionTransformer(img_size)

        if use_pretrained == True:
            # To load from local disk
            # ckpt_path = PathManager.GetBasePath() + "dinov2_vits14_pretrain.pth"

            # ckpt = torch.load(ckpt_path, map_location="cpu")

            # To load from server
            ckpt = torch.hub.load_state_dict_from_url(
                "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
                map_location="cpu"
            )

            filtered_state = {}
            for k, v in ckpt.items():
                if k.startswith("head") or k == "pos_embed":
                    continue
                filtered_state[k] = v

            missing, unexpected = self.semantics_encoder.load_state_dict(filtered_state, strict=False)

            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        if freeze_dino == True:
            # Freeze all DINO parameters
            for param in self.semantics_encoder.parameters():
                param.requires_grad = False

        self.conv_rgb = nn.Conv2d(in_channels=3, out_channels=num_feats,
                                   kernel_size=kernel_size, padding=1)
        self.conv_depth = nn.Conv2d(in_channels=1, out_channels=num_feats,
                                  kernel_size=kernel_size, padding=1)
        

        # GTA blocks
        self.GTA1 = GTA(num_feats, 1, kernel_size)
        self.GTA2 = GTA(num_feats, 2, kernel_size)
        self.GTA3 = GTA(num_feats, 2, kernel_size)
        self.GTA4 = GTA(num_feats, 3, kernel_size)

        # Featur fusion blocks
        self.bridge1 = FeatureFusion(channels=num_feats, rgb_channels=num_feats,scale=scale)
        self.bridge2 = FeatureFusion(channels=2*num_feats, rgb_channels=num_feats,scale=scale)
        self.bridge3 = FeatureFusion(channels=3*num_feats, rgb_channels=num_feats,scale=scale)


        self.bicubic = nn.Upsample(scale_factor=scale, mode='bicubic')

        # Resizing layers between GTA blocks
        self.conv1x1 = ConvOps.DefaultConv(num_feats, num_feats, 1)
        self.conv1x1_2 = ConvOps.DefaultConv(2*num_feats, 2*num_feats, 1)
        self.conv1x1_3 = ConvOps.DefaultConv(2*num_feats, 3*num_feats, 1)

        self.reshaper = ConvOps.DefaultConv(4*num_feats, 2*num_feats, 1)
        self.reshaper2 = ConvOps.DefaultConv(8*num_feats, 3*num_feats, 1)

        
        self.conv_proj = nn.Conv2d(in_channels=num_feats, out_channels=2*num_feats,
                                  kernel_size=kernel_size, padding=1)

        # Default activation
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Upsampler
        self.upsampler = UpsamplerBlock(num_feats, kernel_size, scale)

    def forward(self, image, depth):
        # 1. Get DINO semnatic tokens
        features = self.semantics_encoder.get_intermediate_layers(image, self.intermediate_layer_idx, return_class_token=True)

        H = image.shape[-2] 
        W = image.shape[-1] 
       
        # 2. Init features for raw RGB and LR depth
        depth1 = self.act(self.conv_depth(depth))
        rgb1 = self.act(self.conv_rgb(image))

        # GTA Block 1
        dp_enc1, rgb_enc1 = self.GTA1(depth1, features[0][0], rgb1, H, W)        
        dp_enc_proj1 = self.conv1x1(
            dp_enc1
        )
        # Fuse and generate next features
        ca1_in, rgb2 = self.bridge1(dp_enc_proj1, rgb_enc1)
        depth2 = torch.cat([dp_enc1, ca1_in + depth1], 1)

        # GTA Block 2
        dp_enc2, rgb_enc2 = self.GTA2(depth2, features[1][0], rgb2, H, W)        
        dp_enc_proj2 = self.conv1x1_2(
            dp_enc2
        )
        # Fuse and generate next features
        ca2_in, rgb3 = self.bridge2(dp_enc_proj2, rgb_enc2)
        depth3 = self.reshaper(torch.cat([dp_enc2, ca2_in + self.conv_proj(depth1)], 1))

        # GTA Block 3
        dp_enc3, rgb_enc3 = self.GTA3(depth3, features[2][0], rgb3, H, W)        
        dp_enc_proj3 = self.conv1x1_3(
            dp_enc3
        )
        # Fuse and generate next features
        ca3_in, rgb4 = self.bridge3(dp_enc_proj3, rgb_enc3)
        depth4 = ca3_in


        # GTA Block 4
        dp_enc4, rgb_enc4 = self.GTA4(depth4, features[3][0], rgb4, H, W)
        
        # Pass through the upsampler, discarded RGB

        out = self.upsampler(dp_enc4)
        return out + self.bicubic(depth)