import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tifffile import imread
from torch.utils.data import DataLoader, Dataset


DEFAULT_CONFIG = {
    "data_root": "DATA/eGFP-LaminB3",
    "image_suffix": "_n2v_3d.tif",
    "mask_suffix": "_anno.tif",
    "model_dir": "models",
    "model_name": "unet3d_nuclei",
    "patch_size": [32, 256, 256],
    "batch_size": 2,
    "patches_per_vol": 20,
    "val_fraction": 0.2,
    "augment": True
}


def load_config(config_path: str) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        user_config = json.load(f)
    config = dict(DEFAULT_CONFIG)
    config.update(user_config)
    return config


def find_pairs(data_root: Path, image_suffix: str, mask_suffix: str):
    masks = sorted(data_root.rglob(f"*{mask_suffix}"))
    pairs = []

    for mask_path in masks:
        stem = mask_path.name[: -len(mask_suffix)]
        img_path = mask_path.with_name(stem + image_suffix)
        if img_path.exists():
            pairs.append((img_path, mask_path))
        else:
            print(f"missing image for mask: {mask_path}")

    return pairs


def load_volume_pair(img_path: Path, mask_path: Path):
    img = np.asarray(imread(img_path), dtype=np.float32)
    mask = np.asarray(imread(mask_path))

    if img.shape != mask.shape:
        raise ValueError(f"Shape mismatch for {img_path.name}: image {img.shape}, mask {mask.shape}")

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    mask = (mask > 0).astype(np.float32)
    return img, mask


def split_pairs(pairs, val_fraction: float = 0.2):
    if len(pairs) <= 1:
        return pairs, []

    n_val = max(1, int(round(len(pairs) * val_fraction)))
    n_val = min(n_val, len(pairs) - 1)
    return pairs[:-n_val], pairs[-n_val:]


class NucleiDataset3D(Dataset):
    def __init__(self, volumes, patch_size, patches_per_vol=20, augment=False):
        self.volumes = volumes
        self.pz, self.py, self.px = patch_size
        self.patches_per_vol = patches_per_vol
        self.augment = augment
        self.samples = []

        for vol_idx, (img, msk) in enumerate(self.volumes):
            z_max, y_max, x_max = img.shape
            attempts = 0
            found = 0

            while found < self.patches_per_vol and attempts < self.patches_per_vol * 100:
                attempts += 1
                z0 = np.random.randint(0, max(1, z_max - self.pz + 1))
                y0 = np.random.randint(0, max(1, y_max - self.py + 1))
                x0 = np.random.randint(0, max(1, x_max - self.px + 1))

                patch_msk = msk[z0:z0 + self.pz, y0:y0 + self.py, x0:x0 + self.px]
                if patch_msk.shape != (self.pz, self.py, self.px):
                    continue
                if patch_msk.max() > 0:
                    self.samples.append((vol_idx, z0, y0, x0))
                    found += 1

            print(f"volume {vol_idx}: sampled {found} patches")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vol_idx, z0, y0, x0 = self.samples[idx]
        img, msk = self.volumes[vol_idx]

        patch_img = img[z0:z0 + self.pz, y0:y0 + self.py, x0:x0 + self.px]
        patch_msk = msk[z0:z0 + self.pz, y0:y0 + self.py, x0:x0 + self.px]

        if self.augment:
            patch_img, patch_msk = augment_patch(patch_img, patch_msk)

        return (
            torch.from_numpy(patch_img).unsqueeze(0),
            torch.from_numpy(patch_msk).unsqueeze(0),
        )


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=(16, 32, 64, 128)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)
        self.up_convs = nn.ModuleList()
        self.ups = nn.ModuleList()

        ch = in_ch
        for feat in features:
            self.downs.append(ConvBlock3D(ch, feat))
            ch = feat

        self.bottleneck = ConvBlock3D(ch, ch * 2)
        ch = ch * 2

        for feat in reversed(features):
            self.up_convs.append(nn.ConvTranspose3d(ch, feat, kernel_size=2, stride=2))
            self.ups.append(ConvBlock3D(feat * 2, feat))
            ch = feat

        self.head = nn.Conv3d(ch, out_ch, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up_conv, up, skip in zip(self.up_convs, self.ups, reversed(skips)):
            x = up_conv(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = up(torch.cat([skip, x], dim=1))

        return self.head(x)


def dice_loss(logits, target, smooth=1.0):
    pred = torch.sigmoid(logits).reshape(-1)
    target = target.reshape(-1)
    inter = (pred * target).sum()
    return 1 - (2 * inter + smooth) / (pred.sum() + target.sum() + smooth)


def bce_dice_loss(logits, target, pos_weight):
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    return bce + dice_loss(logits, target)


def augment_patch(img, mask):
    if np.random.rand() < 0.5:
        img = img[:, :, ::-1]
        mask = mask[:, :, ::-1]
    if np.random.rand() < 0.5:
        img = img[:, ::-1, :]
        mask = mask[:, ::-1, :]
    if np.random.rand() < 0.5:
        k = np.random.randint(1, 4)
        img = np.rot90(img, k=k, axes=(1, 2))
        mask = np.rot90(mask, k=k, axes=(1, 2))

    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.9, 1.1)
        shift = np.random.uniform(-0.05, 0.05)
        img = np.clip(img * scale + shift, 0.0, 1.0)

    if np.random.rand() < 0.3:
        noise = np.random.normal(0.0, 0.02, size=img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)

    return np.ascontiguousarray(img, dtype=np.float32), np.ascontiguousarray(mask, dtype=np.float32)


def evaluate(model, dataloader, device, pos_weight):
    model.eval()
    losses = []
    with torch.no_grad():
        for imgs, msks in dataloader:
            imgs = imgs.to(device)
            msks = msks.to(device)
            logits = model(imgs)
            losses.append(bce_dice_loss(logits, msks, pos_weight).item())
    return float(np.mean(losses)) if losses else np.nan


def main():
    parser = argparse.ArgumentParser(description="Train a 3D U-Net on *_n2v_3d.tif inputs and *_anno.tif masks.")
    parser.add_argument("--config", default=None,
                        help="Optional JSON config with data_root, suffixes, patch_size, and batch_size")
    parser.add_argument("--data_root", default=None,
                        help="Root directory to recursively scan for training pairs")
    parser.add_argument("--image_suffix", default=None,
                        help="Suffix for input image volumes")
    parser.add_argument("--mask_suffix", default=None,
                        help="Suffix for annotation mask volumes")
    parser.add_argument("--model_dir", default=None,
                        help="Directory where the trained model will be saved")
    parser.add_argument("--model_name", default=None,
                        help="Output model name")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Training learning rate")
    parser.add_argument("--patch_z", type=int, default=None,
                        help="Patch size in Z")
    parser.add_argument("--patch_y", type=int, default=None,
                        help="Patch size in Y")
    parser.add_argument("--patch_x", type=int, default=None,
                        help="Patch size in X")
    parser.add_argument("--patches_per_vol", type=int, default=None,
                        help="Number of positive patches to sample per volume")
    parser.add_argument("--val_fraction", type=float, default=None,
                        help="Fraction of volume pairs to use for validation")
    parser.add_argument("--gpu", action="store_true",
                        help="Use CUDA if available")
    args = parser.parse_args()
    config = load_config(args.config) if args.config else dict(DEFAULT_CONFIG)

    data_root = Path(args.data_root or config["data_root"])
    image_suffix = args.image_suffix or config["image_suffix"]
    mask_suffix = args.mask_suffix or config["mask_suffix"]
    model_dir = Path(args.model_dir or config["model_dir"])
    model_name = args.model_name or config["model_name"]
    batch_size = args.batch_size or int(config["batch_size"])
    patches_per_vol = args.patches_per_vol or int(config["patches_per_vol"])
    val_fraction = args.val_fraction if args.val_fraction is not None else float(config["val_fraction"])
    augment = bool(config.get("augment", True))
    patch_cfg = config["patch_size"]
    patch_size = (
        args.patch_z or int(patch_cfg[0]),
        args.patch_y or int(patch_cfg[1]),
        args.patch_x or int(patch_cfg[2]),
    )

    pairs = find_pairs(data_root, image_suffix, mask_suffix)
    if not pairs:
        raise FileNotFoundError(
            f"No training pairs found under {data_root} with image suffix {image_suffix} and mask suffix {mask_suffix}"
        )

    train_pairs, val_pairs = split_pairs(pairs, val_fraction=val_fraction)
    print(f"total pairs : {len(pairs)}")
    print(f"train pairs : {len(train_pairs)}")
    print(f"val pairs   : {len(val_pairs)}")

    train_vols = [load_volume_pair(img_path, mask_path) for img_path, mask_path in train_pairs]
    val_vols = [load_volume_pair(img_path, mask_path) for img_path, mask_path in val_pairs] if val_pairs else []

    train_ds = NucleiDataset3D(
        train_vols,
        patch_size=patch_size,
        patches_per_vol=patches_per_vol,
        augment=augment,
    )
    val_ds = NucleiDataset3D(
        val_vols,
        patch_size=patch_size,
        patches_per_vol=max(4, patches_per_vol // 2),
        augment=False,
    ) if val_vols else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False) if val_ds is not None and len(val_ds) > 0 else None

    print(f"train patches: {len(train_ds)}")
    if val_ds is not None:
        print(f"val patches  : {len(val_ds)}")
    print(f"patch size   : {patch_size}")
    print(f"batch size   : {batch_size}")
    print(f"augment      : {augment}")

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    model = UNet3D().to(device)
    print(f"device       : {device}")
    print(f"parameters   : {sum(p.numel() for p in model.parameters()):,}")

    all_masks = torch.cat([msk.unsqueeze(0) for _, msk in train_ds], dim=0)
    fg = all_masks.mean().item()
    pos_weight = torch.tensor([(1 - fg) / (fg + 1e-6)], device=device)
    print(f"foreground fraction: {fg:.4f}")
    print(f"pos_weight         : {pos_weight.item():.2f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)

    train_losses = []
    val_losses = []
    best_val = np.inf

    out_dir = model_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{model_name}.pth"
    best_model_path = out_dir / f"{model_name}_best.pth"
    plot_path = out_dir / f"{model_name}_loss.png"

    print("Starting 3D U-Net training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for imgs, msks in train_loader:
            imgs = imgs.to(device)
            msks = msks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = bce_dice_loss(logits, msks, pos_weight)
            loss.backward()
            optimizer.step()
            running += loss.item()

        avg_train = running / max(1, len(train_loader))
        train_losses.append(avg_train)

        avg_val = evaluate(model, val_loader, device, pos_weight) if val_loader is not None else np.nan
        val_losses.append(avg_val)
        scheduler.step(avg_val if not np.isnan(avg_val) else avg_train)

        if val_loader is not None and avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), best_model_path)

        if epoch == 1 or epoch % 5 == 0:
            if np.isnan(avg_val):
                print(f"epoch {epoch:3d}/{args.epochs}  train={avg_train:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")
            else:
                print(
                    f"epoch {epoch:3d}/{args.epochs}  train={avg_train:.4f}  "
                    f"val={avg_val:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}"
                )

    torch.save(model.state_dict(), model_path)

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="train")
    if val_loader is not None:
        plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()

    print(f"Saved model      : {model_path}")
    if best_val < np.inf:
        print(f"Saved best model : {best_model_path}")
    print(f"Saved loss plot  : {plot_path}")


if __name__ == "__main__":
    main()
