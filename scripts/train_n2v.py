"""
Train a simple 3D Noise2Void model on nuclei volumes from Zeiss CZI files.

Examples
--------
python train_n2v.py \
  --image_path "DATA/eGFP-LaminB3/ST22/260304_ID468_1024x1024_1zoom_1umz__ST22_GFPlaminB3_laminB1ab_embryo2_1-5zoom_-01.czi" \
  --model_name n2v_nuclei_3d_single

python train_n2v.py \
  --data_root "DATA/eGFP-LaminB3" \
  --val_image_count 2 \
  --model_name n2v_nuclei_3d_all
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from n2v.models import N2V, N2VConfig
from czifile import imread as czi_imread


DATA_ROOT = Path("DATA/eGFP-LaminB3")
MODEL_DIR = Path("models")
DEFAULT_MODEL_NAME = "n2v_nuclei_3d"
DEFAULT_CONFIG = {
    "data_root": "DATA/eGFP-LaminB3",
    "glob": "*.czi",
    "skip_stems": ["settings"],
    "channel": 0,
    "model_name": "n2v_nuclei_3d",
    "model_mode": "3d",
    "layout": [0, 0, "channel", 0, "z", "y", "x", 0],
}


def load_config(config_path: str) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        user_config = json.load(f)
    config = dict(DEFAULT_CONFIG)
    config.update(user_config)
    return config


def discover_czi_files(data_root: str, glob_pattern: str, skip_stems) -> List[Path]:
    skip = {stem.lower() for stem in skip_stems}
    return sorted(
        p for p in Path(data_root).rglob(glob_pattern)
        if p.suffix.lower() == ".czi" and p.stem.lower() not in skip
    )


def load_volume_from_config(image_path: Path, config: dict) -> np.ndarray:
    image = np.asarray(czi_imread(str(image_path)), dtype=np.float32)
    layout = config["layout"]
    channel = int(config.get("channel", 0))

    if image.ndim != len(layout):
        raise ValueError(
            f"Config layout length {len(layout)} does not match image ndim {image.ndim} "
            f"for {image_path.name}, shape={image.shape}"
        )

    index = []
    kept_axes = []
    for token in layout:
        if token == "channel":
            index.append(channel)
        elif token in {"z", "y", "x"}:
            index.append(slice(None))
            kept_axes.append(token)
        else:
            index.append(int(token))

    volume = np.asarray(image[tuple(index)], dtype=np.float32)
    if volume.ndim != len(kept_axes):
        raise ValueError(f"Unexpected extracted shape {volume.shape} with kept axes {kept_axes}")

    axis_map = {name: i for i, name in enumerate(kept_axes)}
    if set(kept_axes) == {"z", "y", "x"}:
        volume = np.transpose(volume, [axis_map["z"], axis_map["y"], axis_map["x"]])
    elif kept_axes == ["y", "x"]:
        volume = volume[np.newaxis, ...]
    else:
        raise ValueError(f"Layout must preserve either z/y/x or y/x, got {kept_axes}")

    return np.squeeze(volume).astype(np.float32)


def percentile_normalize(img: np.ndarray, pmin: float = 1.0, pmax: float = 99.8) -> np.ndarray:
    lo = np.percentile(img, pmin)
    hi = np.percentile(img, pmax)
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    img = np.clip(img, lo, hi)
    return ((img - lo) / (hi - lo)).astype(np.float32)


def center_crop_xy(volume: np.ndarray, crop_size: Optional[int]) -> np.ndarray:
    if crop_size is None:
        return volume

    z, h, w = volume.shape
    crop = min(crop_size, h, w)
    y0 = (h - crop) // 2
    x0 = (w - crop) // 2
    return volume[:, y0:y0 + crop, x0:x0 + crop]


def limit_z(volume: np.ndarray, max_z: Optional[int]) -> np.ndarray:
    if max_z is None or volume.shape[0] <= max_z:
        return volume

    z0 = (volume.shape[0] - max_z) // 2
    return volume[z0:z0 + max_z]


def build_3d_dataset(image_paths, config, crop_size: Optional[int], max_z: Optional[int]) -> np.ndarray:
    """Return array shaped (N, Z, Y, X, 1)."""
    volumes = []

    for image_path in image_paths:
        volume = load_volume_from_config(image_path, config)
        volume = percentile_normalize(volume)
        volume = limit_z(volume, max_z=max_z)
        volume = center_crop_xy(volume, crop_size=crop_size)

        if np.std(volume) <= 1e-6:
            print(f"{image_path.name}: skipped empty volume")
            continue

        volumes.append(volume[..., np.newaxis])
        print(f"{image_path.name}: kept volume {volume.shape}")

    if not volumes:
        raise ValueError("No non-empty nuclei volumes found for training.")

    return np.stack(volumes, axis=0)


def split_image_paths(image_paths, val_image_count: int):
    if len(image_paths) <= 1:
        return image_paths, []

    val_image_count = min(val_image_count, len(image_paths) - 1)
    train_paths = image_paths[:-val_image_count]
    val_paths = image_paths[-val_image_count:]
    return train_paths, val_paths


def build_config(
    train_data: np.ndarray,
    patch_size_xy: int,
    patch_size_z: int,
    batch_size: int,
    epochs: int,
    steps_per_epoch: int,
):
    return N2VConfig(
        train_data,
        unet_kern_size=3,
        train_steps_per_epoch=steps_per_epoch,
        train_epochs=epochs,
        train_loss="mse",
        batch_norm=True,
        train_batch_size=batch_size,
        n2v_perc_pix=0.198,
        n2v_patch_shape=(patch_size_z, patch_size_xy, patch_size_xy),
        n2v_manipulator="uniform_withCP",
        n2v_neighborhood_radius=5,
    )


def main():
    parser = argparse.ArgumentParser(description="Train a clean 3D Noise2Void model on nuclei CZI data.")
    parser.add_argument("--config", required=True,
                        help="Path to JSON config describing data_root, channel, and CZI z/y/x layout")
    parser.add_argument("--image_path", default=None,
                        help="Single CZI file to train from")
    parser.add_argument("--data_root", default=str(DATA_ROOT),
                        help="Root folder to recursively scan for .czi files when --image_path is not given")
    parser.add_argument("--model_dir", default=str(MODEL_DIR),
                        help="Directory where the trained model will be saved")
    parser.add_argument("--model_name", default=None,
                        help="Model name inside model_dir")
    parser.add_argument("--val_image_count", type=int, default=1,
                        help="Number of full images to hold out for validation")
    parser.add_argument("--crop_size", type=int, default=256,
                        help="Center crop size in XY before training; set 0 to disable")
    parser.add_argument("--max_z", type=int, default=32,
                        help="Maximum Z depth to use from the center of each volume; set 0 to disable")
    parser.add_argument("--patch_size", type=int, default=64,
                        help="XY patch size for 3D N2V")
    parser.add_argument("--patch_size_z", type=int, default=16,
                        help="Z patch size for 3D N2V")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=100,
                        help="Training steps per epoch")
    args = parser.parse_args()
    config = load_config(args.config)
    model_name = args.model_name or config.get("model_name") or DEFAULT_MODEL_NAME

    if args.image_path:
        image_paths = [Path(args.image_path)]
    else:
        data_root = args.data_root if args.data_root != str(DATA_ROOT) else config["data_root"]
        image_paths = discover_czi_files(
            data_root=data_root,
            glob_pattern=config.get("glob", "*.czi"),
            skip_stems=config.get("skip_stems", []),
        )

    if not image_paths:
        raise FileNotFoundError("No .czi files found for training.")

    crop_size = None if args.crop_size == 0 else args.crop_size
    max_z = None if args.max_z == 0 else args.max_z

    train_paths, val_paths = split_image_paths(image_paths, args.val_image_count)

    print(f"Training from {len(image_paths)} CZI file(s)")
    print(f"train images: {len(train_paths)}")
    print(f"val images: {len(val_paths)}")

    x_train = build_3d_dataset(train_paths, config=config, crop_size=crop_size, max_z=max_z)
    x_val = build_3d_dataset(val_paths, config=config, crop_size=crop_size, max_z=max_z) if val_paths else x_train[:1]

    print(f"train volumes: {len(x_train)}")
    print(f"val volumes: {len(x_val)}")
    print(f"sample shape: {x_train.shape[1:]}")

    config = build_config(
        x_train,
        patch_size_xy=args.patch_size,
        patch_size_z=args.patch_size_z,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
    )

    model = N2V(config=config, name=model_name, basedir=args.model_dir)
    history = model.train(x_train, x_val)

    print(f"Saved model to: {Path(args.model_dir) / model_name}")
    print(f"History keys: {list(history.history.keys())}")


if __name__ == "__main__":
    main()
