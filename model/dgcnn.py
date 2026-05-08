import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcloud_core import (
    BasePointCloudAutoEncoder,
    MLPDecoder,
    encode_numpy,
    pairwise_squared_distance,
    reconstruct_numpy,
)


def knn_feature_indices(x, k):
    """
    Find k nearest neighbors for each point in feature space.

    Args:
        x: (B, C, N)
        k: int

    Returns:
        idx: (B, N, k)
    """
    points = x.transpose(1, 2).contiguous()  # (B, N, C)
    dist2 = pairwise_squared_distance(points)  # (B, N, N)

    _, num_points, _ = dist2.shape
    diag_mask = torch.eye(
        num_points,
        device=x.device,
        dtype=torch.bool,
    ).unsqueeze(0)
    dist2 = dist2.masked_fill(diag_mask, float("inf"))

    k = min(k, num_points - 1)
    if k < 1:
        raise ValueError("DGCNN requires point clouds with at least 2 points.")

    idx = dist2.topk(k=k, dim=-1, largest=False)[1]
    return idx


def get_graph_feature(x, k, idx=None):
    """
    Build local edge features used by DGCNN.

    Args:
        x: (B, C, N)
        k: int
        idx: optional neighbor indices with shape (B, N, k)

    Returns:
        edge_feature: (B, 2 * C, N, k)
    """
    batch_size, num_dims, num_points = x.shape

    if idx is None:
        idx = knn_feature_indices(x, k=k)

    idx_base = torch.arange(batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).reshape(-1)

    points = x.transpose(1, 2).contiguous()  # (B, N, C)
    neighbors = points.reshape(batch_size * num_points, num_dims)[idx, :]
    neighbors = neighbors.view(batch_size, num_points, -1, num_dims)

    centers = points.unsqueeze(2).expand(-1, -1, neighbors.shape[2], -1)
    edge_feature = torch.cat((neighbors - centers, centers), dim=-1)
    edge_feature = edge_feature.permute(0, 3, 1, 2).contiguous()
    return edge_feature


class DGCNNEncoder(nn.Module):
    """
    DGCNN encoder that maps a point cloud to a global latent vector.

    Input:
        x: (B, N, 3)

    Output:
        z: (B, latent_dim)
    """
    def __init__(self, latent_dim=128, k=20, emb_dims=1024, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.k = k
        self.emb_dims = emb_dims

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(emb_dims * 2, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError("Input to DGCNNEncoder must have shape (B, N, 3).")

        num_points = x.shape[1]
        if num_points < 2:
            raise ValueError("DGCNNEncoder requires at least 2 points per cloud.")

        k = min(self.k, num_points - 1)
        x = x.transpose(1, 2).contiguous()  # (B, 3, N)

        x1 = self.conv1(get_graph_feature(x, k=k)).max(dim=-1)[0]
        x2 = self.conv2(get_graph_feature(x1, k=k)).max(dim=-1)[0]
        x3 = self.conv3(get_graph_feature(x2, k=k)).max(dim=-1)[0]
        x4 = self.conv4(get_graph_feature(x3, k=k)).max(dim=-1)[0]

        features = torch.cat((x1, x2, x3, x4), dim=1)
        features = self.conv5(features)

        global_max = F.adaptive_max_pool1d(features, 1).squeeze(-1)
        global_avg = F.adaptive_avg_pool1d(features, 1).squeeze(-1)
        global_feat = torch.cat((global_max, global_avg), dim=1)

        z = self.fc(global_feat)
        return z


class DGCNNPointCloudAutoEncoder(BasePointCloudAutoEncoder):
    """
    DGCNN encoder + MLP decoder autoencoder.
    """
    def __init__(
        self,
        latent_dim=128,
        num_points=256,
        k=20,
        emb_dims=1024,
        encoder_dropout=0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_points = num_points
        self.k = k
        self.emb_dims = emb_dims
        self.encoder_dropout = encoder_dropout

        self.encoder = DGCNNEncoder(
            latent_dim=latent_dim,
            k=k,
            emb_dims=emb_dims,
            dropout=encoder_dropout,
        )
        self.decoder = MLPDecoder(latent_dim=latent_dim, num_points=num_points)
        self.training_history = None


class PointCloudAutoEncoder(DGCNNPointCloudAutoEncoder):
    """
    Compatibility alias so the DGCNN variant can be imported with the same
    class name as the PointNet autoencoder.
    """


__all__ = [
    "DGCNNEncoder",
    "DGCNNPointCloudAutoEncoder",
    "PointCloudAutoEncoder",
    "encode_numpy",
    "reconstruct_numpy",
    "get_graph_feature",
    "knn_feature_indices",
]
