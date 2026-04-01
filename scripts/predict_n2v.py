"""
Run Noise2Void prediction on one CZI file or on all CZI files under a folder.

The denoised TIFF is written next to each source .czi file.

Examples
--------
python predect.py \
  --model_name n2v_nuclei_3d_all \
  --model_mode 3d \
  --data_root "DATA/eGFP-LaminB3"

python predect.py \
  --model_name n2v_nuclei_3d_all \
  --model_mode 3d \
  --image_path "DATA/eGFP-LaminB3/ST22/example.czi"
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from czifile import imread as czi_imread
from n2v.models import N2V
from tifffile import imwrite


MODEL_DIR = Path("models")
DATA_ROOT = Path("DATA/eGFP-LaminB3")
N_TILES_2D = (2, 2)
N_TILES_3D = (2, 4, 4, 1)
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


def discover_czi_files(data_root: str, glob_pattern: str, skip_stems):
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


def predict_2d(model: N2V, volume: np.ndarray) -> np.ndarray:
    pred_slices = []
    for z in range(volume.shape[0]):
        pred = model.predict(volume[z], axes="YX", n_tiles=N_TILES_2D)
        pred_slices.append(pred.astype(np.float32))
    return np.stack(pred_slices, axis=0)


def predict_3d(model: N2V, volume: np.ndarray, n_tiles_3d) -> np.ndarray:
    volume_zyxc = volume[..., np.newaxis]
    pred = model.predict(volume_zyxc, axes="ZYXC", n_tiles=n_tiles_3d)
    return np.squeeze(pred, axis=-1).astype(np.float32)


def predict_volume(model: N2V, volume: np.ndarray, model_mode: str, n_tiles_3d) -> np.ndarray:
    if model_mode == "2d":
        return predict_2d(model, volume)
    if model_mode == "3d":
        return predict_3d(model, volume, n_tiles_3d=n_tiles_3d)
    raise ValueError("model_mode must be '2d' or '3d'")


def output_path_for(image_path: Path, model_mode: str) -> Path:
    return image_path.with_name(f"{image_path.stem}_n2v_{model_mode}.tif")


def main():
    parser = argparse.ArgumentParser(description="Predict denoised nuclei volumes from CZI files using Noise2Void.")
    parser.add_argument("--config", required=True,
                        help="Path to JSON config describing data_root, channel, and CZI z/y/x layout")
    parser.add_argument("--model_dir", default=str(MODEL_DIR),
                        help="Directory containing the trained N2V model")
    parser.add_argument("--model_name", default=None,
                        help="Model name inside model_dir")
    parser.add_argument("--model_mode", choices=["2d", "3d"], default=None,
                        help="Prediction mode matching the trained model")
    parser.add_argument("--data_root", default=str(DATA_ROOT),
                        help="Root folder to recursively scan for .czi files when --image_path is not given")
    parser.add_argument("--image_path", default=None,
                        help="Single .czi file to predict")
    parser.add_argument("--crop_size", type=int, default=0,
                        help="Optional center crop in XY before prediction; 0 disables")
    parser.add_argument("--max_z", type=int, default=0,
                        help="Optional center crop in Z before prediction; 0 disables")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing *_n2v_*.tif outputs")
    args = parser.parse_args()
    config = load_config(args.config)
    model_name = args.model_name or config.get("model_name")
    model_mode = args.model_mode or config.get("model_mode", "3d")
    if not model_name:
        raise ValueError("model_name must be provided either in the config JSON or via --model_name")
    crop_size = args.crop_size or config.get("predict_crop_size") or config.get("crop_size") or 0
    max_z = args.max_z or config.get("predict_max_z") or config.get("max_z") or 0
    crop_size = None if crop_size == 0 else int(crop_size)
    max_z = None if max_z == 0 else int(max_z)
    n_tiles_3d = tuple(config.get("n_tiles_3d", N_TILES_3D))

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
        raise FileNotFoundError("No .czi files found for prediction.")

    model = N2V(config=None, name=model_name, basedir=str(args.model_dir))

    print(f"Model      : {Path(args.model_dir) / model_name}")
    print(f"Mode       : {model_mode}")
    print(f"CZI files  : {len(image_paths)}")
    if model_mode == "3d":
        print(f"crop_size  : {crop_size}")
        print(f"max_z      : {max_z}")
        print(f"n_tiles_3d : {n_tiles_3d}")

    for image_path in image_paths:
        out_path = output_path_for(image_path, model_mode)
        if out_path.exists() and not args.overwrite:
            print(f"skip existing: {out_path.name}")
            continue

        print(f"predicting: {image_path}")
        try:
            volume = load_volume_from_config(image_path, config)
            volume_norm = percentile_normalize(volume)
            volume_norm = limit_z(volume_norm, max_z=max_z)
            volume_norm = center_crop_xy(volume_norm, crop_size=crop_size)
            volume_pred = predict_volume(model, volume_norm, model_mode, n_tiles_3d=n_tiles_3d)
        except Exception as exc:
            print(f"skip invalid : {image_path} ({exc})")
            continue

        imwrite(out_path, volume_pred.astype(np.float32))
        print(f"saved     : {out_path}")


if __name__ == "__main__":
    main()
