import torch
from torch import nn
import torch.nn.functional as F

class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


class MultiHeadSelfAttention(nn.Module) :
    def __init__(self, pose_embed_dim):
        super().__init__()
        self.MHSA = nn.MultiheadAttention(embed_dim = pose_embed_dim, num_heads = 8, batch_first=True)
        self.LayerNorm1 = nn.LayerNorm(pose_embed_dim)
        self.FC = nn.Sequential(nn.Linear(pose_embed_dim, pose_embed_dim), nn.Dropout(0.05), nn.GELU())
        self.LayerNorm2 = nn.LayerNorm(pose_embed_dim)

    def forward(self, x) :
        x_skip = x
        x, _ = self.MHSA(x, x, x)
        x = x + x_skip
        x = self.LayerNorm1(x)
        x_skip = x
        x = self.FC(x) + x_skip
        x = self.LayerNorm2(x)

        return x