"""
pointcloud.py
-------------
Build point clouds from annotation masks using a JSON config.

For every *_anno.tif under the configured read_dir, create a folder next to
that annotation file. The folder is named after the image stem without
'_anno.tif', and contains one .ply point-cloud per label ID.

Example
-------
    python scripts/pointcloud.py --config configs/pointcloud.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from skimage.filters import gaussian
from skimage.measure import marching_cubes
from tifffile import imread


def load_config(config_path: str) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def nucleus_to_pointcloud(mask, label, voxel_size, n_points, smooth=False, smooth_sigma_um=0.1):
    volume = (mask == label).astype(np.float32)

    if smooth:
        sigma_vox = np.asarray(smooth_sigma_um, dtype=np.float32) / np.asarray(voxel_size, dtype=np.float32)
        volume = gaussian(volume, sigma=sigma_vox, preserve_range=True)

    verts, faces, _, _ = marching_cubes(volume, level=0.5, spacing=voxel_size)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    pts -= pts.mean(axis=0)
    return pts


def output_dir_for_anno(anno_path: Path, anno_suffix: str) -> Path:
    name = anno_path.name
    if name.endswith(anno_suffix):
        stem = name[:-len(anno_suffix)]
    else:
        stem = anno_path.stem
    return anno_path.parent / stem


def process_anno(anno_path: Path, anno_suffix: str, voxel_size, n_points, min_voxels, smooth, smooth_sigma_um):
    mask = np.asarray(imread(anno_path))
    labels = np.unique(mask)
    labels = labels[labels > 0]
    labels = [lbl for lbl in labels if (mask == lbl).sum() >= min_voxels]

    out_dir = output_dir_for_anno(anno_path, anno_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{anno_path} -> {out_dir}")
    print(f"  cells to process: {len(labels)}")
    print(f"  smoothing: {'on' if smooth else 'off'}")
    print(f"  voxel size: {tuple(voxel_size)}")

    for lbl in labels:
        pts = nucleus_to_pointcloud(
            mask=mask,
            label=lbl,
            voxel_size=voxel_size,
            n_points=n_points,
            smooth=smooth,
            smooth_sigma_um=smooth_sigma_um,
        )
        trimesh.PointCloud(vertices=pts).export(out_dir / f"{lbl}.ply")


def main():
    parser = argparse.ArgumentParser(description="Build point clouds from *_anno.tif files using a JSON config.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config with read_dir, anno_suffix, smoothing, and optional voxel_size_um"
    )
    parser.add_argument("--read_dir", default=None, help="Optional override for input root directory")
    parser.add_argument("--smooth", action="store_true", help="Enable smoothing regardless of config")
    parser.add_argument("--no-smooth", action="store_true", help="Disable smoothing regardless of config")
    args = parser.parse_args()

    config = load_config(args.config)

    read_dir = Path(args.read_dir or config["read_dir"])
    anno_suffix = config["anno_suffix"]
    voxel_size = np.asarray(config.get("voxel_size_um", [1.0, 1.0, 1.0]), dtype=np.float32)
    n_points = int(config["n_points"])
    min_voxels = int(config["min_voxels"])
    smooth = bool(config.get("smooth", False))

    if args.smooth:
        smooth = True
    if args.no_smooth:
        smooth = False

    smooth_sigma_um = float(config.get("smooth_sigma_um", 0.1))

    anno_files = sorted(read_dir.rglob(f"*{anno_suffix}"))
    if not anno_files:
        raise FileNotFoundError(f"No annotation files matching *{anno_suffix} found under {read_dir}")

    print(f"Found {len(anno_files)} annotation file(s) in {read_dir}")
    for anno_path in anno_files:
        process_anno(
            anno_path=anno_path,
            anno_suffix=anno_suffix,
            voxel_size=voxel_size,
            n_points=n_points,
            min_voxels=min_voxels,
            smooth=smooth,
            smooth_sigma_um=smooth_sigma_um,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
