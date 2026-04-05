import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TRGRAMKQV(nn.Module):
    def __init__(self, channels, t_channels):
        super(TRGRAMKQV, self).__init__()
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(t_channels, channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(t_channels, channels, kernel_size=1, bias=False)

        self.alpha = nn.Parameter(torch.tensor(0.0))

        self._init_weights()

    def _init_weights(self):
        # Xavier is best for attention projections
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1.0)

        # Start as identity
        nn.init.zeros_(self.alpha)

    def compute_simple_attention(self, q, k, v):
        """
        q, k, v: [B, C, H, W]
        """
        B, C, H, W = q.shape
        N = H * W

        # Flatten spatial dims
        q = q.view(B, C, N).transpose(1, 2)  # [B, N, C]
        k = k.view(B, C, N)                  # [B, C, N]
        v = v.view(B, C, N).transpose(1, 2)  # [B, N, C]

        # Optional: normalization for stability
        q = F.normalize(q, dim=-1, eps=1e-6)
        k = F.normalize(k, dim=1, eps=1e-6)

        # Spatial attention
        attn = torch.bmm(q, k)               # [B, N, N]
        attn = attn / math.sqrt(C)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v)             # [B, N, C]
        out = out.transpose(1, 2).view(B, C, H, W)

        return out

    def compute_gram_qkv(self, q, k, v):
        """
        q, k, v: [B, C, H, W]
        """
        B, C, H, W = q.shape
        N = H * W

        # Flatten spatial dims
        q = q.view(B, C, N)
        k = k.view(B, C, N)
        v = v.view(B, C, N)

        # Center + normalize (critical for stability)
        q = q - q.mean(dim=-1, keepdim=True)
        k = k - k.mean(dim=-1, keepdim=True)

        q = F.normalize(q, dim=-1, eps=1e-6)
        k = F.normalize(k, dim=-1, eps=1e-6)

        # Channel attention (Gram-style)
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / math.sqrt(N)
        attn = F.softmax(attn, dim=-1)

        # Apply to values
        out = torch.bmm(attn, v)
        return out.view(B, C, H, W)

    def forward(self, x, token):
        q = self.q_proj(x)
        k = self.k_proj(token)
        v = self.v_proj(token)

        gram_residual = self.compute_simple_attention(q, k, v)
        return x + self.alpha * gram_residual