"""
pointcloud_mari.py
------------------
Build point clouds from timelapse cellpose mask tifs using a track CSV.

For every mask file in mask_dir (named t{frame}_*.tif), the script:
  1. Looks up tracked centroids for that frame from the track CSV.
  2. Finds the mask label at each centroid voxel.
  3. Samples n_points from the nucleus surface voxels (fast) or via
     marching-cubes (method="marching_cubes", slow but smoother).
  4. Saves each nucleus as {track_id}_t{frame}_{label}_{cell_type}.ply.
  5. Writes metadata.csv and track_sub.csv under out_dir.

Only nuclei whose cell_type != filter_type (default "unknown") are processed.
Mask files are processed in parallel across CPU cores.

Example
-------
    python scripts/pointcloud_mari.py --config configs/pointcloud_mari.json
"""

import argparse
import json
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from tifffile import imread


def load_config(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_t(stem: str) -> int:
    m = re.search(r't(\d+)', stem)
    if m is None:
        raise ValueError(f"Cannot parse time index from filename: {stem}")
    return int(m.group(1))


def surface_voxels(binary_vol):
    """Return voxel coords that are on the surface (at least one background neighbor)."""
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(binary_vol)
    surface = binary_vol & ~eroded
    return np.argwhere(surface)


def nucleus_to_pointcloud_voxel(mask, label, voxel_size, n_points, slc=None):
    if slc is None:
        from scipy.ndimage import find_objects
        objects = find_objects(mask == label)
        if not objects or objects[0] is None:
            raise ValueError(f"label {label} not found in mask")
        slc = objects[0]
    binary = (mask[slc] == label)
    coords = surface_voxels(binary).astype(np.float32)
    if coords.size == 0:
        raise ValueError(f"label {label} has no surface voxels")
    offset = np.array([slc[0].start, slc[1].start, slc[2].start], dtype=np.float32)
    coords += offset
    coords *= np.asarray(voxel_size, dtype=np.float32)
    idx = np.random.choice(len(coords), size=n_points, replace=len(coords) < n_points)
    pts = coords[idx]
    pts -= pts.mean(axis=0)
    return pts


def nucleus_to_pointcloud_mc(mask, label, voxel_size, n_points, slc=None):
    from skimage.measure import marching_cubes
    if slc is None:
        from scipy.ndimage import find_objects
        objects = find_objects(mask == label)
        if not objects or objects[0] is None:
            raise ValueError(f"label {label} not found in mask")
        slc = objects[0]
    volume = (mask[slc] == label).astype(np.float32)
    verts, faces, _, _ = marching_cubes(volume, level=0.5, spacing=voxel_size)
    offset = np.array([slc[0].start, slc[1].start, slc[2].start], dtype=np.float32)
    verts += offset * np.asarray(voxel_size, dtype=np.float32)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)
    pts -= pts.mean(axis=0)
    return pts


def process_mask_file(args):
    mf, track_rows, out_dir, cfg = args

    voxel_size = np.asarray(cfg["voxel_size_um"], dtype=np.float32)
    n_points   = cfg["n_points"]
    min_voxels = cfg["min_voxels"]
    method     = cfg.get("method", "voxel")
    col_track_id = cfg["col_track_id"]
    col_type     = cfg["col_type"]
    col_z, col_y, col_x = cfg["col_z"], cfg["col_y"], cfg["col_x"]
    col_annotation = cfg.get("col_annotation")

    t = parse_t(Path(mf).stem)
    mask = np.asarray(imread(mf))
    from scipy.ndimage import find_objects
    # one bounding-box pass for all labels in this frame
    label_slices = find_objects(mask)

    zs = track_rows[col_z].to_numpy()
    ys = track_rows[col_y].to_numpy()
    xs = track_rows[col_x].to_numpy()

    in_bounds = (
        (zs >= 0) & (zs < mask.shape[0]) &
        (ys >= 0) & (ys < mask.shape[1]) &
        (xs >= 0) & (xs < mask.shape[2])
    )
    track_rows = track_rows[in_bounds]
    zs, ys, xs = zs[in_bounds], ys[in_bounds], xs[in_bounds]
    labels = mask[zs, ys, xs]

    # Build one table with track row + sampled label at centroid.
    work_df = track_rows.copy()
    work_df["_z"] = zs
    work_df["_y"] = ys
    work_df["_x"] = xs
    work_df["_label"] = labels.astype(np.int64)
    work_df = work_df[work_df["_label"] != 0]
    if work_df.empty:
        print(f"  t={t}: 0 nuclei saved")
        return []

    # Keep only one row per label for point-cloud generation (fast path),
    # then re-attach all rows for metadata/export naming.
    rep = work_df.drop_duplicates(subset="_label", keep="first").copy()
    label_ids = rep["_label"].to_numpy(dtype=np.int64)
    unique_labels, counts = np.unique(mask, return_counts=True)
    label_sizes = {int(lbl): int(cnt) for lbl, cnt in zip(unique_labels, counts)}
    valid_labels = {int(lbl) for lbl in label_ids if label_sizes.get(int(lbl), 0) >= min_voxels}
    if not valid_labels:
        print(f"  t={t}: 0 nuclei saved")
        return []

    points_by_label = {}
    for label in sorted(valid_labels):
        try:
            slc = label_slices[label - 1] if 0 < label <= len(label_slices) else None
            if slc is None:
                continue
            if method == "marching_cubes":
                pts = nucleus_to_pointcloud_mc(mask, label, voxel_size, n_points, slc=slc)
            else:
                pts = nucleus_to_pointcloud_voxel(mask, label, voxel_size, n_points, slc=slc)
            points_by_label[label] = pts
        except Exception as e:
            print(f"  ERROR t={t} label={label}: {e}")

    records = []
    for idx, row in work_df.iterrows():
        label = int(row["_label"])
        pts = points_by_label.get(label)
        if pts is None:
            continue
        track_id = row[col_track_id]
        cell_type = row[col_type]
        z, y, x = int(row["_z"]), int(row["_y"]), int(row["_x"])

        ply_name = f"{track_id}_t{t}_{label}_{cell_type}.ply"
        trimesh.PointCloud(vertices=pts).export(Path(out_dir) / ply_name)
        records.append({
            "ply_file":  ply_name,
            "track_id":  track_id,
            "frame":     t,
            "label":     label,
            "cell_type": cell_type,
            "annotation": row[col_annotation] if col_annotation and col_annotation in row.index else None,
            col_z: z, col_y: y, col_x: x,
        })

    print(f"  t={t}: {len(records)} nuclei saved")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    mask_dir     = Path(cfg.get("read_dir", cfg.get("mask_dir")))
    track_csv    = Path(cfg["track_csv"])
    out_dir      = Path(cfg["out_dir"])
    anno_suffix  = cfg.get("anno_suffix", ".tif")
    filter_type  = cfg.get("filter_type", "unknown")
    col_frame    = cfg.get("col_frame",    "Spot frame")
    col_track_id = cfg.get("col_track_id", "TRACK_ID")
    col_type     = cfg.get("col_type",     "cell_type")
    col_annotation = cfg.get("col_annotation", None)
    col_z        = cfg.get("col_z",        "Z_orig")
    col_y        = cfg.get("col_y",        "Y_orig")
    col_x        = cfg.get("col_x",        "X_orig")

    # bake column names into cfg dict for worker access
    cfg.update({"col_track_id": col_track_id, "col_type": col_type,
                "col_z": col_z, "col_y": col_y, "col_x": col_x,
                "col_annotation": col_annotation})

    out_dir.mkdir(parents=True, exist_ok=True)

    track = pd.read_csv(track_csv)
    track = track[track[col_type] != filter_type].copy()
    track[col_frame] = track[col_frame].astype(int)
    track[col_z] = track[col_z].round().astype(int)
    track[col_y] = track[col_y].round().astype(int)
    track[col_x] = track[col_x].round().astype(int)
    track_by_t = {t: grp for t, grp in track.groupby(col_frame)}

    mask_files = sorted(mask_dir.rglob(f"*{anno_suffix}"))
    print(f"Found {len(mask_files)} mask files | method: {cfg.get('method','voxel')}")

    tasks = [
        (str(mf), track_by_t[parse_t(mf.stem)], str(out_dir), cfg)
        for mf in mask_files
        if parse_t(mf.stem) in track_by_t
    ]

    n_workers = args.workers or cpu_count()
    print(f"Processing {len(tasks)} frames with {n_workers} workers...")

    with Pool(n_workers) as pool:
        results = pool.map(process_mask_file, tasks)

    records = [r for batch in results for r in batch]
    meta_df = pd.DataFrame(records)
    meta_df.to_csv(out_dir / "metadata.csv", index=False)

    # sub-table: original track rows that were successfully saved
    t_values = {parse_t(mf.stem) for mf in mask_files}
    track_sub = track[track[col_frame].isin(t_values)].merge(
        meta_df[["track_id", "frame", "label", "ply_file"]],
        left_on=[col_track_id, col_frame],
        right_on=["track_id", "frame"],
        how="inner",
    ).drop(columns=["track_id", "frame"])
    track_sub.to_csv(out_dir / "track_sub.csv", index=False)

    print(f"\nDone. {len(records)} point clouds saved to {out_dir}")
    print(f"Metadata:  {out_dir / 'metadata.csv'}")
    print(f"Track sub: {out_dir / 'track_sub.csv'}")


if __name__ == "__main__":
    main()
