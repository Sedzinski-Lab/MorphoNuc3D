"""
pointcloud.py
-------------
For every *_anno.tif in ANNO_DIR, create a sub-folder and write one
.ply point-cloud file per segmented cell (named by label ID).

Usage
-----
    python pointcloud.py                          # uses defaults below
    python pointcloud.py --read_dir path/to/anno --save_dir path/to/out
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from skimage.filters import gaussian
from skimage.measure import marching_cubes
from tifffile import imread

# ── Defaults (match notebook) ─────────────────────────────────────────────
ANNO_DIR   = '260227_ID459_laminB1antibody/anno'
VOXEL_SIZE = np.array([0.3, 0.14, 0.14])   # µm per voxel (Z, Y, X)
N_POINTS   = 1024                           # surface samples per cell
MIN_VOXELS = 100                             # skip fragments smaller than this
SMOOTH_SIGMA_UM = 0.14                      # mild default smoothing in physical units (µm)


# ── Core function ─────────────────────────────────────────────────────────
def nucleus_to_pointcloud(mask, label, voxel_size, n_points, smooth=False, smooth_sigma_um=SMOOTH_SIGMA_UM):
    """Return (n_points, 3) µm surface points centered at origin."""
    volume = (mask == label).astype(np.float32)
    if smooth:
        # Convert physical sigma (µm) to per-axis voxel sigma (Z, Y, X).
        sigma_vox = np.asarray(smooth_sigma_um, dtype=np.float32) / voxel_size
        volume = gaussian(volume, sigma=sigma_vox, preserve_range=True)

    verts, faces, _, _ = marching_cubes(volume, level=0.5, spacing=voxel_size)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    pts -= pts.mean(axis=0)   # center at origin
    return pts


def process_anno(anno_path: Path, out_root: Path, smooth=False, smooth_sigma_um=SMOOTH_SIGMA_UM):
    """Process one *_anno.tif → folder of per-cell .ply files."""
    # folder name: strip _anno.tif suffix
    name = anno_path.name.replace('_anno.tif', '')
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    mask   = imread(anno_path)                      # (Z, Y, X) instance labels
    labels = np.unique(mask)
    labels = labels[labels > 0]                     # remove background

    # filter tiny fragments
    labels = [l for l in labels if (mask == l).sum() >= MIN_VOXELS]

    print(f'\n{anno_path.name}  →  {out_dir}')
    print(f'  cells to process: {len(labels)}')
    if smooth:
        print(f'  smoothing: on (sigma={smooth_sigma_um} µm)')
    else:
        print('  smoothing: off')

    for lbl in labels:
        pts = nucleus_to_pointcloud(
            mask, lbl, VOXEL_SIZE, N_POINTS,
            smooth=smooth, smooth_sigma_um=smooth_sigma_um
        )
        cloud = trimesh.PointCloud(vertices=pts)
        cloud.export(out_dir / f'{lbl}.ply')

    print(f'  saved {len(labels)} .ply files')


# ── Entry point ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Build per-cell point clouds from annotation masks.')
    parser.add_argument('--read_dir', '--anno_dir', dest='anno_dir', default=ANNO_DIR,
                        help='Directory containing *_anno.tif files (input/read dir)')
    parser.add_argument('--save_dir', '--out_dir', dest='out_dir', default=None,
                        help='Root output directory for .ply files (default: same as read_dir)')
    parser.add_argument('--smooth', action='store_true',
                        help='Enable mild Gaussian smoothing before marching cubes')
    parser.add_argument('--smooth_sigma_um', type=float, default=SMOOTH_SIGMA_UM,
                        help='Smoothing sigma in microns when --smooth is enabled')
    args = parser.parse_args()

    anno_dir = Path(args.anno_dir)
    out_root = Path(args.out_dir) if args.out_dir else anno_dir

    anno_files = sorted(anno_dir.glob('*_anno.tif'))
    if not anno_files:
        print(f'No *_anno.tif files found in {anno_dir}')
        return

    print(f'Found {len(anno_files)} annotation file(s) in {anno_dir}')
    for f in anno_files:
        process_anno(
            f, out_root,
            smooth=args.smooth,
            smooth_sigma_um=args.smooth_sigma_um
        )

    print('\nDone.')


if __name__ == '__main__':
    main()
