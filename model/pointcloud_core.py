import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


_UPPER_TRIANGLE_INDEX_CACHE = {}


def to_tensor(x, device=None, dtype=torch.float32):
    """
    Convert input to torch.Tensor.

    Supported input:
    - numpy array with shape (N, 3) or (B, N, 3)
    - torch tensor with shape (N, 3) or (B, N, 3)
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif not torch.is_tensor(x):
        raise TypeError("Input must be a numpy array or a torch tensor.")

    x = x.to(dtype=dtype)

    if x.ndim == 2:
        # (N, 3) -> (1, N, 3)
        x = x.unsqueeze(0)

    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError("Input must have shape (N, 3) or (B, N, 3).")

    if device is not None:
        x = x.to(device)

    return x


def resolve_device(module, device=None):
    """
    Resolve the target device for a module.
    """
    if device is not None:
        return torch.device(device)

    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def squeeze_if_single_batch(x):
    """
    Remove the batch axis only for single-sample outputs.
    """
    if x.shape[0] == 1:
        return x.squeeze(0)
    return x


def pairwise_squared_distance(x, y=None):
    """
    Compute squared pairwise distances.

    Args:
        x: (B, N, C)
        y: (B, M, C) or None

    Returns:
        dist2: (B, N, M)
    """
    if y is None:
        y = x

    x2 = (x ** 2).sum(dim=-1, keepdim=True)           # (B, N, 1)
    y2 = (y ** 2).sum(dim=-1).unsqueeze(1)            # (B, 1, M)
    xy = torch.bmm(x, y.transpose(1, 2))              # (B, N, M)
    dist2 = x2 + y2 - 2.0 * xy
    return torch.clamp(dist2, min=0.0)


def safe_sqrt(x):
    """
    Numerically stable sqrt with zero gradient at exactly zero.
    """
    return torch.where(x > 0, torch.sqrt(x), torch.zeros_like(x))


def get_upper_triangle_indices(n, device):
    """
    Cache upper-triangle indices for repeated all-pairs computations.
    """
    device = torch.device(device)
    key = (n, device.type, device.index)
    if key not in _UPPER_TRIANGLE_INDEX_CACHE:
        _UPPER_TRIANGLE_INDEX_CACHE[key] = torch.triu_indices(
            n,
            n,
            offset=1,
            device=device,
        )
    return _UPPER_TRIANGLE_INDEX_CACHE[key]


def chamfer_distance(x, y, reduction="mean"):
    """
    Symmetric Chamfer distance between two point clouds.

    Args:
        x: (B, N, 3)
        y: (B, M, 3)
        reduction: "mean" or "sum"

    Returns:
        scalar loss
    """
    dist2 = pairwise_squared_distance(x, y)  # (B, N, M)

    min_x_to_y = dist2.min(dim=2)[0]         # (B, N)
    min_y_to_x = dist2.min(dim=1)[0]         # (B, M)

    loss_per_batch = min_x_to_y.mean(dim=1) + min_y_to_x.mean(dim=1)  # (B,)

    if reduction == "mean":
        return loss_per_batch.mean()
    elif reduction == "sum":
        return loss_per_batch.sum()
    else:
        raise ValueError("reduction must be 'mean' or 'sum'.")


def knn_indices(x, k):
    """
    Find k nearest neighbors for each point inside the same point cloud.

    Args:
        x: (B, N, 3)
        k: int

    Returns:
        idx: (B, N, k)
    """
    dist2 = pairwise_squared_distance(x)  # (B, N, N)

    # Exclude self-neighbor by setting diagonal to a large value.
    _, N, _ = dist2.shape
    device = x.device
    diag_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)  # (1, N, N)
    dist2 = dist2.masked_fill(diag_mask, float("inf"))

    idx = dist2.topk(k=k, dim=-1, largest=False)[1]  # (B, N, k)
    return idx


def gather_knn_points(x, idx):
    """
    Gather neighbor points by indices.

    Args:
        x: (B, N, C)
        idx: (B, N, k)

    Returns:
        neighbors: (B, N, k, C)
    """
    B, N, C = x.shape
    k = idx.shape[-1]

    idx_expanded = idx.unsqueeze(-1).expand(B, N, k, C)
    x_expanded = x.unsqueeze(1).expand(B, N, N, C)
    neighbors = torch.gather(x_expanded, dim=2, index=idx_expanded)
    return neighbors


def relative_distance_loss(
    x_true,
    x_pred,
    mode="global",
    k=16,
    num_pairs=2048,
    p=1,
):
    """
    Relative geometry preservation loss.

    Two modes are supported:

    1) mode="global"
       Compare sampled pairwise distances between all points.

    2) mode="knn"
       Build kNN graph on x_true and compare local edge lengths
       between x_true and x_pred using the same neighbor indices.

    Args:
        x_true: (B, N, 3)
        x_pred: (B, N, 3)
        mode: "global" or "knn"
        k: number of neighbors for knn mode
        num_pairs: number of sampled pairs for global mode.
            Use `None` or `"all"` to compare all unique point pairs.
        p: 1 or 2, use L1 or L2 difference on distances

    Returns:
        scalar loss
    """
    if x_true.shape != x_pred.shape:
        raise ValueError("x_true and x_pred must have the same shape.")

    B, N, _ = x_true.shape
    device = x_true.device

    if mode == "global":
        if num_pairs is None or num_pairs == "all":
            if N < 2:
                raise ValueError("Global all-pairs mode requires at least 2 points.")

            # Select unique off-diagonal pairs before sqrt so diagonal zero
            # distances never enter the gradient path.
            pair_i, pair_j = get_upper_triangle_indices(N, device)
            dist2_true = pairwise_squared_distance(x_true)[:, pair_i, pair_j]
            dist2_pred = pairwise_squared_distance(x_pred)[:, pair_i, pair_j]

            d_true = safe_sqrt(dist2_true)
            d_pred = safe_sqrt(dist2_pred)
        else:
            if not isinstance(num_pairs, int) or num_pairs <= 0:
                raise ValueError("num_pairs must be a positive integer, None, or 'all'.")

            # Randomly sample point pairs to avoid building the full N x N matrix when N is large.
            i_idx = torch.randint(0, N, (B, num_pairs), device=device)
            j_idx = torch.randint(0, N, (B, num_pairs), device=device)

            x_true_i = torch.gather(x_true, 1, i_idx.unsqueeze(-1).expand(-1, -1, 3))
            x_true_j = torch.gather(x_true, 1, j_idx.unsqueeze(-1).expand(-1, -1, 3))
            x_pred_i = torch.gather(x_pred, 1, i_idx.unsqueeze(-1).expand(-1, -1, 3))
            x_pred_j = torch.gather(x_pred, 1, j_idx.unsqueeze(-1).expand(-1, -1, 3))

            d_true = torch.norm(x_true_i - x_true_j, dim=-1)  # (B, num_pairs)
            d_pred = torch.norm(x_pred_i - x_pred_j, dim=-1)  # (B, num_pairs)

        if p == 1:
            loss = torch.abs(d_true - d_pred).mean()
        elif p == 2:
            loss = ((d_true - d_pred) ** 2).mean()
        else:
            raise ValueError("p must be 1 or 2.")

        return loss

    elif mode == "knn":
        idx = knn_indices(x_true, k=k)  # (B, N, k)

        true_neighbors = gather_knn_points(x_true, idx)  # (B, N, k, 3)
        pred_neighbors = gather_knn_points(x_pred, idx)  # (B, N, k, 3)

        x_true_center = x_true.unsqueeze(2)  # (B, N, 1, 3)
        x_pred_center = x_pred.unsqueeze(2)  # (B, N, 1, 3)

        d_true = torch.norm(true_neighbors - x_true_center, dim=-1)  # (B, N, k)
        d_pred = torch.norm(pred_neighbors - x_pred_center, dim=-1)  # (B, N, k)

        if p == 1:
            loss = torch.abs(d_true - d_pred).mean()
        elif p == 2:
            loss = ((d_true - d_pred) ** 2).mean()
        else:
            raise ValueError("p must be 1 or 2.")

        return loss

    else:
        raise ValueError("mode must be 'global' or 'knn'.")


class MLPDecoder(nn.Module):
    """
    MLP decoder that reconstructs a fixed number of points.

    Input:
        z: (B, latent_dim)

    Output:
        x_recon: (B, num_points, 3)
    """
    def __init__(self, latent_dim=128, num_points=256):
        super().__init__()
        self.num_points = num_points

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_points * 3),
        )

    def forward(self, z):
        x = self.net(z)                            # (B, num_points * 3)
        x = x.view(z.shape[0], self.num_points, 3)
        return x


class ReconstructionLoss(nn.Module):
    """
    Combined reconstruction loss:
    - Chamfer distance
    - Relative distance preservation loss
    """
    def __init__(
        self,
        chamfer_weight=1.0,
        relative_weight=1.0,
        relative_mode="knn",
        knn_k=16,
        global_num_pairs=2048,
        relative_p=1,
    ):
        super().__init__()
        self.chamfer_weight = chamfer_weight
        self.relative_weight = relative_weight
        self.relative_mode = relative_mode
        self.knn_k = knn_k
        self.global_num_pairs = global_num_pairs
        self.relative_p = relative_p

    def forward(self, x_true, x_pred):
        zero = x_true.new_zeros(())

        if self.chamfer_weight == 0:
            loss_chamfer = zero
        else:
            loss_chamfer = chamfer_distance(x_true, x_pred)

        if self.relative_weight == 0:
            loss_relative = zero
        else:
            loss_relative = relative_distance_loss(
                x_true=x_true,
                x_pred=x_pred,
                mode=self.relative_mode,
                k=self.knn_k,
                num_pairs=self.global_num_pairs,
                p=self.relative_p,
            )

        total = (
            self.chamfer_weight * loss_chamfer +
            self.relative_weight * loss_relative
        )

        return {
            "loss": total,
            "chamfer": loss_chamfer,
            "relative": loss_relative,
        }


class PointCloudDataset(Dataset):
    """
    Lightweight dataset wrapper for point clouds with shape (B, N, 3).
    """
    def __init__(self, points):
        if isinstance(points, dict):
            if "points" not in points:
                raise KeyError("Dictionary input must contain a 'points' key.")
            points = points["points"]

        self.points = to_tensor(points, dtype=torch.float32).cpu().contiguous()

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, index):
        return self.points[index]


def train_one_epoch(model, dataloader, optimizer, criterion, device="cpu"):
    """
    Train for one epoch.

    Dataloader should yield point clouds with shape (B, N, 3).
    """
    model.train()
    total_loss = 0.0
    total_chamfer = 0.0
    total_relative = 0.0
    n_batches = 0

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            if len(batch) != 1:
                raise ValueError("Expected a single tensor batch from the dataloader.")
            batch = batch[0]

        x = to_tensor(batch, device=device)

        optimizer.zero_grad()
        x_recon, _ = model(x)
        losses = criterion(x, x_recon)
        losses["loss"].backward()
        optimizer.step()

        total_loss += losses["loss"].item()
        total_chamfer += losses["chamfer"].item()
        total_relative += losses["relative"].item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "chamfer": total_chamfer / max(n_batches, 1),
        "relative": total_relative / max(n_batches, 1),
    }


class BasePointCloudAutoEncoder(nn.Module):
    """
    Shared training and inference API for point-cloud autoencoders.

    Subclasses provide `self.encoder`, `self.decoder`, `self.num_points`, and
    `self.training_history` in __init__.
    """
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def train_model(
        self,
        points,
        epochs=50,
        batch_size=8,
        lr=1e-3,
        device=None,
        shuffle=True,
        num_workers=0,
        weight_decay=0.0,
        chamfer_weight=1.0,
        relative_weight=1.0,
        relative_mode="knn",
        knn_k=16,
        global_num_pairs=2048,
        relative_p=1,
        verbose=True,
    ):
        """
        Train the autoencoder on a batch of point clouds.

        Args:
            points:
                numpy array or torch tensor with shape (B, N, 3).
                A dictionary containing a `points` key is also accepted.
        """
        dataset = PointCloudDataset(points)
        input_num_points = dataset.points.shape[1]

        if input_num_points != self.num_points:
            raise ValueError(
                f"Input point clouds have {input_num_points} points, "
                f"but the model decoder expects {self.num_points}. "
                f"Initialize {self.__class__.__name__}(num_points=...) "
                "with a matching value."
            )

        if relative_mode == "knn":
            knn_k = min(knn_k, input_num_points - 1)
            if knn_k < 1:
                raise ValueError("kNN relative loss requires point clouds with at least 2 points.")

        device = resolve_device(self, device)
        self.to(device)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = ReconstructionLoss(
            chamfer_weight=chamfer_weight,
            relative_weight=relative_weight,
            relative_mode=relative_mode,
            knn_k=knn_k,
            global_num_pairs=global_num_pairs,
            relative_p=relative_p,
        )

        history = {
            "epoch": [],
            "loss": [],
            "chamfer": [],
            "relative": [],
        }

        for epoch_idx in range(epochs):
            metrics = train_one_epoch(
                model=self,
                dataloader=dataloader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )

            history["epoch"].append(epoch_idx + 1)
            history["loss"].append(metrics["loss"])
            history["chamfer"].append(metrics["chamfer"])
            history["relative"].append(metrics["relative"])

            if verbose:
                print(
                    f"Epoch {epoch_idx + 1:03d}/{epochs:03d} | "
                    f"loss={metrics['loss']:.6f} | "
                    f"chamfer={metrics['chamfer']:.6f} | "
                    f"relative={metrics['relative']:.6f}"
                )

        self.training_history = history
        return history

    @torch.no_grad()
    def encode(self, x, device=None):
        """
        Encode one or more point clouds into latent vectors.
        """
        self.eval()
        device = resolve_device(self, device)
        x = to_tensor(x, device=device)
        z = self.encoder(x)
        return squeeze_if_single_batch(z).cpu().numpy()

    @torch.no_grad()
    def reconstruct(self, x, device=None):
        """
        Reconstruct one or more point clouds.
        """
        self.eval()
        device = resolve_device(self, device)
        x = to_tensor(x, device=device)
        x_recon, z = self(x)
        return (
            squeeze_if_single_batch(x_recon).cpu().numpy(),
            squeeze_if_single_batch(z).cpu().numpy(),
        )

    @torch.no_grad()
    def encode_points(self, x, device=None):
        """
        Backward-compatible alias for `encode`.
        """
        return self.encode(x, device=device)

    @torch.no_grad()
    def reconstruct_points(self, x, device=None):
        """
        Backward-compatible alias for `reconstruct`.
        """
        return self.reconstruct(x, device=device)


def ensure_point_cloud_array(points):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Each point cloud must have shape (N, C>=3), got {points.shape}")
    return points


def resample_point_cloud(points, n_points, rng=None):
    points = ensure_point_cloud_array(points)
    if len(points) == n_points:
        return points
    rng = rng or np.random
    replace = len(points) < n_points
    idx = rng.choice(len(points), n_points, replace=replace)
    return points[idx]


def normalize_point_cloud(points):
    points = ensure_point_cloud_array(points).copy()
    xyz = points[:, :3]
    xyz = xyz - xyz.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(xyz, axis=1))
    if scale > 1e-8:
        xyz = xyz / scale
    points[:, :3] = xyz
    return points


def augment_point_cloud_z_rotation_jitter(points, rng=None, jitter_std=0.01):
    points = ensure_point_cloud_array(points).copy()
    rng = rng or np.random
    xyz = points[:, :3]

    theta = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    xyz = xyz @ rot.T

    jitter = rng.normal(0.0, jitter_std, size=xyz.shape).astype(np.float32)
    points[:, :3] = xyz + jitter
    return points


def canonical_nucleus_label(label):
    if isinstance(label, str):
        x = label.strip().lower()
        if x in {"normal"}:
            return "normal"
        if x in {"ring deformed", "ring_deformed", "ring-deformed", "ring"}:
            return "ring deformed"
        if x in {"point deformed", "point_deformed", "point-deformed", "point"}:
            return "point deformed"
        return x
    return label


def build_nucleus_label_mapping(labels):
    canonical_labels = [canonical_nucleus_label(l) for l in labels]
    preferred_order = ["normal", "ring deformed", "point deformed"]
    unique = list(dict.fromkeys(canonical_labels))

    if all(isinstance(x, (int, np.integer)) for x in unique):
        return {int(x): int(x) for x in sorted(unique)}

    ordered = [x for x in preferred_order if x in unique]
    ordered += [x for x in unique if x not in ordered]
    return {lab: i for i, lab in enumerate(ordered)}


@torch.no_grad()
def encode_numpy(model, x_np, device="cpu"):
    """
    Encode a single N x 3 numpy array into a latent vector.
    """
    return model.encode(x_np, device=device)


@torch.no_grad()
def reconstruct_numpy(model, x_np, device="cpu"):
    """
    Reconstruct a single N x 3 numpy array.
    """
    return model.reconstruct(x_np, device=device)


__all__ = [
    "BasePointCloudAutoEncoder",
    "MLPDecoder",
    "PointCloudDataset",
    "ReconstructionLoss",
    "augment_point_cloud_z_rotation_jitter",
    "build_nucleus_label_mapping",
    "canonical_nucleus_label",
    "chamfer_distance",
    "encode_numpy",
    "ensure_point_cloud_array",
    "gather_knn_points",
    "get_upper_triangle_indices",
    "knn_indices",
    "normalize_point_cloud",
    "pairwise_squared_distance",
    "reconstruct_numpy",
    "relative_distance_loss",
    "resample_point_cloud",
    "resolve_device",
    "safe_sqrt",
    "squeeze_if_single_batch",
    "to_tensor",
    "train_one_epoch",
]
