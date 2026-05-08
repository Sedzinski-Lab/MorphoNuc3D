import copy
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pointcloud_core import (
    augment_point_cloud_z_rotation_jitter,
    build_nucleus_label_mapping,
    canonical_nucleus_label,
    normalize_point_cloud,
    pairwise_squared_distance,
    resample_point_cloud,
)


class NucleusPointCloudDataset(Dataset):
    def __init__(
        self,
        point_clouds,
        labels=None,
        n_points=512,
        normalize=True,
        augment=False,
        label_to_idx=None,
    ):
        self.point_clouds = [np.asarray(pc, dtype=np.float32) for pc in point_clouds]
        self.labels = labels
        self.n_points = n_points
        self.normalize = normalize
        self.augment = augment

        if labels is not None:
            if label_to_idx is None:
                label_to_idx = self._build_label_mapping(labels)
            self.label_to_idx = label_to_idx
            self.encoded_labels = np.array(
                [self.label_to_idx[self._canonical_label(l)] for l in labels],
                dtype=np.int64,
            )
        else:
            self.label_to_idx = label_to_idx
            self.encoded_labels = None

    @staticmethod
    def _canonical_label(label):
        return canonical_nucleus_label(label)

    def _build_label_mapping(self, labels):
        return build_nucleus_label_mapping(labels)

    def __len__(self):
        return len(self.point_clouds)

    def _resample_points(self, points):
        return resample_point_cloud(points, self.n_points)

    def _normalize_points(self, points):
        return normalize_point_cloud(points)

    def _augment_points(self, points):
        return augment_point_cloud_z_rotation_jitter(points)

    def __getitem__(self, idx):
        points = self.point_clouds[idx]

        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Each point cloud must have shape (N, C>=3), got {points.shape}")

        points = self._resample_points(points)

        if self.normalize:
            points = self._normalize_points(points)

        if self.augment:
            points = self._augment_points(points)

        points = torch.tensor(points, dtype=torch.float32)

        if self.encoded_labels is None:
            return points

        label = torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        return points, label


class PointFeatureEncoder(nn.Module):
    def __init__(self, input_dim=3, feature_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, feature_dim, kernel_size=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, N, C]
        x = x.transpose(1, 2)            # [B, C, N]
        z = self.net(x)                  # [B, D, N]
        z = z.transpose(1, 2)            # [B, N, D]
        return z


class PointMILNet(nn.Module):
    def __init__(
        self,
        input_dim=3,
        num_classes=3,
        feature_dim=256,
        pooling="conjunctive",
        context_k=0,
        dropout=0.1,
    ):
        super().__init__()
        assert pooling in {"instance", "attention", "additive", "conjunctive"}
        self.pooling = pooling
        self.context_k = context_k
        self.num_classes = num_classes

        self.encoder = PointFeatureEncoder(
            input_dim=input_dim,
            feature_dim=feature_dim,
            dropout=dropout,
        )

        self.attn_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.instance_cls_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self.bag_cls_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def _smooth_attention(self, attn, xyz):
        if self.context_k <= 0 or xyz.shape[1] <= 1:
            return attn

        k = min(self.context_k, xyz.shape[1])
        dist2 = pairwise_squared_distance(xyz)  # [B, N, N]
        knn_idx = dist2.topk(k=k, largest=False).indices  # [B, N, k]

        attn_flat = attn.squeeze(-1)  # [B, N]
        expanded = attn_flat.unsqueeze(1).expand(-1, xyz.shape[1], -1)  # [B, N, N]
        smoothed = torch.gather(expanded, 2, knn_idx).mean(dim=-1, keepdim=True)  # [B, N, 1]
        return smoothed

    def forward(self, x, return_aux=False):
        # x: [B, N, C]
        xyz = x[:, :, :3]
        z = self.encoder(x)  # [B, N, D]

        if self.pooling == "instance":
            attn = torch.ones(z.shape[0], z.shape[1], 1, device=z.device, dtype=z.dtype)
            point_logits = self.instance_cls_head(z)               # [B, N, K]
            bag_logits = point_logits.mean(dim=1)                  # [B, K]
            point_contrib = F.softmax(point_logits, dim=-1)        # [B, N, K]

        elif self.pooling == "attention":
            attn = self.attn_head(z)                               # [B, N, 1]
            attn = self._smooth_attention(attn, xyz)
            pooled = (attn * z).mean(dim=1)                        # [B, D]
            point_logits = None
            bag_logits = self.bag_cls_head(pooled)                 # [B, K]
            point_contrib = attn                                   # [B, N, 1]

        elif self.pooling == "additive":
            attn = self.attn_head(z)                               # [B, N, 1]
            attn = self._smooth_attention(attn, xyz)
            weighted_z = attn * z                                  # [B, N, D]
            point_logits = self.instance_cls_head(weighted_z)      # [B, N, K]
            bag_logits = point_logits.mean(dim=1)                  # [B, K]
            point_contrib = attn * F.softmax(point_logits, dim=-1) # [B, N, K]

        elif self.pooling == "conjunctive":
            attn = self.attn_head(z)                               # [B, N, 1]
            attn = self._smooth_attention(attn, xyz)
            point_logits = self.instance_cls_head(z)               # [B, N, K]
            bag_logits = (attn * point_logits).mean(dim=1)         # [B, K]
            point_contrib = attn * F.softmax(point_logits, dim=-1) # [B, N, K]

        if return_aux:
            return bag_logits, {
                "features": z,
                "attention": attn,
                "point_logits": point_logits,
                "point_contrib": point_contrib,
            }
        return bag_logits


class NucleusPointCloudMILClassifier:
    def __init__(
        self,
        input_dim=3,
        num_classes=3,
        n_points=512,
        feature_dim=256,
        pooling="conjunctive",
        context_k=12,
        lr=1e-3,
        weight_decay=1e-4,
        device=None,
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.n_points = n_points
        self.feature_dim = feature_dim
        self.pooling = pooling
        self.context_k = context_k
        self.lr = lr
        self.weight_decay = weight_decay

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PointMILNet(
            input_dim=input_dim,
            num_classes=num_classes,
            feature_dim=feature_dim,
            pooling=pooling,
            context_k=context_k,
        ).to(self.device)

        self.label_to_idx = None
        self.idx_to_label = None
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def _build_dataloader(
        self,
        point_clouds,
        labels=None,
        batch_size=16,
        shuffle=False,
        augment=False,
    ):
        dataset = NucleusPointCloudDataset(
            point_clouds=point_clouds,
            labels=labels,
            n_points=self.n_points,
            normalize=True,
            augment=augment,
            label_to_idx=self.label_to_idx,
        )

        if labels is not None and self.label_to_idx is None:
            self.label_to_idx = dataset.label_to_idx
            self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )
        return loader

    @staticmethod
    def _accuracy_from_logits(logits, labels):
        preds = logits.argmax(dim=1)
        return (preds == labels).float().mean().item()

    def _run_one_epoch(self, loader, optimizer=None, class_weights=None):
        is_train = optimizer is not None
        self.model.train(is_train)

        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for batch in loader:
            points, labels = batch
            points = points.to(self.device)
            labels = labels.to(self.device)

            if is_train:
                optimizer.zero_grad()

            logits = self.model(points)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += labels.size(0)

        avg_loss = total_loss / max(total_count, 1)
        avg_acc = total_correct / max(total_count, 1)
        return avg_loss, avg_acc

    def fit(
        self,
        train_point_clouds,
        train_labels,
        val_point_clouds=None,
        val_labels=None,
        epochs=50,
        batch_size=16,
        class_weights=None,
        verbose=True,
    ):
        train_loader = self._build_dataloader(
            train_point_clouds,
            train_labels,
            batch_size=batch_size,
            shuffle=True,
            augment=True,
        )

        val_loader = None
        if val_point_clouds is not None and val_labels is not None:
            val_loader = self._build_dataloader(
                val_point_clouds,
                val_labels,
                batch_size=batch_size,
                shuffle=False,
                augment=False,
            )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        best_state = copy.deepcopy(self.model.state_dict())
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_one_epoch(
                train_loader,
                optimizer=optimizer,
                class_weights=class_weights,
            )
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            if val_loader is not None:
                with torch.no_grad():
                    val_loss, val_acc = self._run_one_epoch(
                        val_loader,
                        optimizer=None,
                        class_weights=class_weights,
                    )
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.model.state_dict())

                if verbose:
                    print(
                        f"Epoch {epoch:03d} | "
                        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                    )
            else:
                best_state = copy.deepcopy(self.model.state_dict())
                if verbose:
                    print(
                        f"Epoch {epoch:03d} | "
                        f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
                    )

        self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, point_clouds, batch_size=32):
        loader = self._build_dataloader(
            point_clouds,
            labels=None,
            batch_size=batch_size,
            shuffle=False,
            augment=False,
        )

        self.model.eval()
        probs_all = []

        with torch.no_grad():
            for points in loader:
                points = points.to(self.device)
                logits = self.model(points)
                probs = F.softmax(logits, dim=1)
                probs_all.append(probs.cpu().numpy())

        return np.concatenate(probs_all, axis=0)

    def encode(self, point_clouds, batch_size=32):
        loader = self._build_dataloader(
            point_clouds,
            labels=None,
            batch_size=batch_size,
            shuffle=False,
            augment=False,
        )

        self.model.eval()
        feats_all = []

        with torch.no_grad():
            for points in loader:
                points = points.to(self.device)
                _, aux = self.model(points, return_aux=True)
                z = aux["features"]  # [B, N, D]
                attn = aux["attention"]  # [B, N, 1]
                pooled = (attn * z).mean(dim=1)
                feats_all.append(pooled.cpu().numpy())

        return np.concatenate(feats_all, axis=0)

    def predict(self, point_clouds, batch_size=32, return_labels=True):
        probs = self.predict_proba(point_clouds, batch_size=batch_size)
        pred_idx = probs.argmax(axis=1)

        if return_labels and self.idx_to_label is not None:
            return [self.idx_to_label[int(i)] for i in pred_idx]
        return pred_idx

    def explain(self, point_cloud, target_class=None, normalize_scores=True):
        loader = self._build_dataloader(
            [point_cloud],
            labels=None,
            batch_size=1,
            shuffle=False,
            augment=False,
        )

        self.model.eval()
        with torch.no_grad():
            points = next(iter(loader)).to(self.device)
            logits, aux = self.model(points, return_aux=True)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))

        if target_class is None:
            target_idx = pred_idx
        elif isinstance(target_class, str):
            target_idx = self.label_to_idx[NucleusPointCloudDataset._canonical_label(target_class)]
        else:
            target_idx = int(target_class)

        if self.pooling == "attention":
            scores = aux["attention"][0, :, 0].detach().cpu().numpy()
        else:
            contrib = aux["point_contrib"][0].detach().cpu().numpy()
            scores = contrib[:, target_idx]

        if normalize_scores:
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        sampled_points = points[0].detach().cpu().numpy()

        return {
            "points": sampled_points,
            "scores": scores,
            "pred_idx": pred_idx,
            "pred_label": self.idx_to_label[pred_idx] if self.idx_to_label is not None else pred_idx,
            "pred_proba": probs,
            "target_idx": target_idx,
            "target_label": self.idx_to_label[target_idx] if self.idx_to_label is not None else target_idx,
        }

    def plot_explanation(
        self,
        explanation,
        elev=20,
        azim=40,
        s=8,
        highlight_q=0.9,
        important_color="red",
        other_color="black",
        other_alpha=0.35,
        important_alpha=0.9,
    ):
        pts = explanation["points"][:, :3]
        scores = explanation["scores"]

        if highlight_q is None:
            mask = np.ones(len(scores), dtype=bool)
        else:
            cutoff = np.quantile(scores, highlight_q)
            mask = scores >= cutoff

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            pts[~mask, 0], pts[~mask, 1], pts[~mask, 2],
            c=other_color,
            s=max(s * 0.8, 1.0),
            alpha=other_alpha,
            linewidths=0,
        )
        ax.scatter(
            pts[mask, 0], pts[mask, 1], pts[mask, 2],
            c=important_color,
            s=max(s * 1.2, 1.0),
            alpha=important_alpha,
            linewidths=0,
        )
        ax.set_title(
            f"Pred: {explanation['pred_label']} | "
            f"Explain: {explanation['target_label']}"
        )
        ax.view_init(elev=elev, azim=azim)
        plt.show()

    def plot_history(self):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].plot(self.history["train_loss"], label="train")
        if len(self.history["val_loss"]) > 0:
            axes[0].plot(self.history["val_loss"], label="val")
        axes[0].set_title("Loss")
        axes[0].legend()

        axes[1].plot(self.history["train_acc"], label="train")
        if len(self.history["val_acc"]) > 0:
            axes[1].plot(self.history["val_acc"], label="val")
        axes[1].set_title("Accuracy")
        axes[1].legend()

        plt.show()
