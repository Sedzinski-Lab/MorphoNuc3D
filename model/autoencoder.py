import torch
import torch.nn as nn

from pointcloud_core import (
    BasePointCloudAutoEncoder,
    MLPDecoder,
    PointCloudDataset,
    ReconstructionLoss,
    chamfer_distance,
    encode_numpy,
    gather_knn_points,
    get_upper_triangle_indices,
    knn_indices,
    pairwise_squared_distance,
    reconstruct_numpy,
    relative_distance_loss,
    resolve_device,
    safe_sqrt,
    squeeze_if_single_batch,
    to_tensor,
    train_one_epoch,
)


class PointNetEncoder(nn.Module):
    """
    Simple PointNet encoder.

    Input:
        x: (B, N, 3)

    Output:
        z: (B, latent_dim)
    """
    def __init__(self, latent_dim=128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x):
        # x: (B, N, 3)
        feat = self.mlp(x)                 # (B, N, 512)
        global_feat = feat.max(dim=1)[0]   # (B, 512), permutation invariant
        z = self.fc(global_feat)           # (B, latent_dim)
        return z


class PointCloudAutoEncoder(BasePointCloudAutoEncoder):
    """
    PointNet encoder + MLP decoder autoencoder.
    """
    def __init__(self, latent_dim=128, num_points=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_points = num_points
        self.encoder = PointNetEncoder(latent_dim=latent_dim)
        self.decoder = MLPDecoder(latent_dim=latent_dim, num_points=num_points)
        self.training_history = None


__all__ = [
    "MLPDecoder",
    "PointCloudAutoEncoder",
    "PointCloudDataset",
    "PointNetEncoder",
    "ReconstructionLoss",
    "chamfer_distance",
    "encode_numpy",
    "gather_knn_points",
    "get_upper_triangle_indices",
    "knn_indices",
    "pairwise_squared_distance",
    "reconstruct_numpy",
    "relative_distance_loss",
    "resolve_device",
    "safe_sqrt",
    "squeeze_if_single_batch",
    "to_tensor",
    "train_one_epoch",
]
