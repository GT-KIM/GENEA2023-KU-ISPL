import torch
from torch import nn
from models.module import MultiHeadSelfAttention


class PoseEmbedder(nn.Module) :
    def __init__(self, config) :
        super().__init__()
        pose_dim = config.model.pose_dim
        pose_embed_dim = config.model.pose_embed_dim

        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(pose_dim, pose_embed_dim), nn.Dropout(0.05), nn.GELU()))
        for _ in range(config.model.pose_embedder_n_layers - 1) :
            self.layers.append(nn.Sequential(nn.Linear(pose_embed_dim, pose_embed_dim), nn.Dropout(0.05), nn.GELU()))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x) :
        x = self.layers(x)
        return x



class MotionEmbedder(nn.Module) :
    def __init__(self, config) :
        super().__init__()
        pose_embed_dim = config.model.pose_embed_dim

        self.RNN = nn.GRU(input_size=pose_embed_dim, hidden_size=pose_embed_dim // 2, num_layers=2, bidirectional=True, batch_first=True)
        self.MHSA = nn.ModuleList()
        for _ in range(config.model.motion_embedder_n_layers) :
            self.MHSA.append(MultiHeadSelfAttention(pose_embed_dim=pose_embed_dim))
        self.MHSA = nn.Sequential(*self.MHSA)

    def forward(self, x) :
        x, _ = self.RNN(x)
        x = self.MHSA(x)
        return x


class PoseDecoder(nn.Module) :
    def __init__(self, config) :
        super().__init__()
        pose_dim = config.model.pose_dim
        pose_embed_dim = config.model.pose_embed_dim

        self.layers = nn.ModuleList()
        for _ in range(config.model.pose_decoder_n_layers - 1):
            self.layers.append(nn.Sequential(nn.Linear(pose_embed_dim, pose_embed_dim), nn.Dropout(0.05), nn.GELU()))
        self.layers.append(nn.Sequential(nn.Linear(pose_embed_dim, pose_dim)))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class MotionDecoder(nn.Module) :
    def __init__(self, config):
        super().__init__()
        pose_embed_dim = config.model.pose_embed_dim

        self.MHSA = nn.ModuleList()
        for _ in range(config.model.motion_decoder_n_layers):
            self.MHSA.append(MultiHeadSelfAttention(pose_embed_dim=pose_embed_dim))
        self.MHSA = nn.Sequential(*self.MHSA)
        self.RNN = nn.GRU(input_size=pose_embed_dim, hidden_size=pose_embed_dim // 2, num_layers=2, bidirectional=True,
                          batch_first=True)

    def forward(self, x):
        x = self.MHSA(x)
        x, _ = self.RNN(x)
        return x

