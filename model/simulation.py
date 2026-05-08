import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

def normalize_rows(x, eps=1e-8):
    """Normalize each row vector to unit length."""
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norm + eps)


def orthonormal_basis_from_axis(axis):
    """Build an orthonormal basis whose first vector is the given axis."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-8)

    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, axis)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    axis_2 = np.cross(axis, ref)
    axis_2 = axis_2 / (np.linalg.norm(axis_2) + 1e-8)
    axis_3 = np.cross(axis, axis_2)
    axis_3 = axis_3 / (np.linalg.norm(axis_3) + 1e-8)
    return axis, axis_2, axis_3


def random_orthonormal_basis(rng):
    """Sample a random right-handed orthonormal basis."""
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q[:, 0], q[:, 1], q[:, 2]


def sample_sphere_point_cloud(n=3000, radius=1.0, noise=0.0, random_state=0):
    """Sample points approximately uniformly on a sphere surface."""
    rng = np.random.default_rng(random_state)
    pts = rng.normal(size=(n, 3))
    pts = normalize_rows(pts)
    pts = radius * pts
    if noise > 0:
        pts += rng.normal(scale=noise, size=pts.shape)
    return pts


def sample_ellipsoid_point_cloud(n=3000, axes=(1.0, 0.8, 0.6), noise=0.0, random_state=0):
    """Sample points on an ellipsoid surface by scaling sphere points."""
    pts = sample_sphere_point_cloud(n=n, radius=1.0, noise=0.0, random_state=random_state)
    pts = pts * np.array(axes)[None, :]
    if noise > 0:
        rng = np.random.default_rng(random_state + 1)
        pts += rng.normal(scale=noise, size=pts.shape)
    return pts


def sample_egg_shape_point_cloud(
    n=3000,
    axes=(1.1, 0.8, 0.8),
    axis=(1.0, 0.0, 0.0),
    basis=None,
    eggness=0.28,
    noise=0.0,
    random_state=0,
):
    """
    Sample a smooth 3D egg shape around a chosen main axis.

    `axes[0]` is the semi-axis length along `axis`. `axes[1]` and `axes[2]`
    are the transverse semi-axis lengths in the two orthogonal directions.
    `eggness > 0` makes the +axis end fuller and the -axis end narrower.
    """
    rng = np.random.default_rng(random_state)

    dirs = rng.normal(size=(n, 3))
    dirs = normalize_rows(dirs)

    if basis is None:
        axis, axis_2, axis_3 = orthonormal_basis_from_axis(axis)
    else:
        basis = np.asarray(basis, dtype=float)
        axis, axis_2, axis_3 = basis

    u1 = dirs @ axis
    u2 = dirs @ axis_2
    u3 = dirs @ axis_3

    # Cross-sections widen smoothly toward the +axis end and narrow toward
    # the opposite end, producing a classic ovoid profile.
    cross_scale = np.clip(1.0 + eggness * u1, 0.2, None)

    x = axes[0] * u1
    y = axes[1] * cross_scale * u2
    z = axes[2] * cross_scale * u3

    pts = (
        x[:, None] * axis[None, :]
        + y[:, None] * axis_2[None, :]
        + z[:, None] * axis_3[None, :]
    )

    if noise > 0:
        pts += rng.normal(scale=noise, size=pts.shape)
    return pts




def center_point_cloud(points):
    """Center point cloud to zero mean."""
    return points - points.mean(axis=0, keepdims=True)


def pca_axis_lengths(points):
    """Estimate shape elongation using PCA standard deviations."""
    pca = PCA(n_components=3)
    pca.fit(points)
    lengths = np.sqrt(pca.explained_variance_)
    return lengths, pca.components_


def simple_shape_summary(points):
    """Return basic geometric summaries."""
    pts = center_point_cloud(points)
    lengths, axes = pca_axis_lengths(pts)
    aspect_ratio_12 = lengths[0] / (lengths[1] + 1e-8)
    aspect_ratio_13 = lengths[0] / (lengths[2] + 1e-8)
    aspect_ratio_23 = lengths[1] / (lengths[2] + 1e-8)
    radius = np.linalg.norm(pts, axis=1)
    return {
        "n_points": len(pts),
        "centroid": pts.mean(axis=0),
        "pca_std": lengths,
        "aspect_ratio_12": aspect_ratio_12,
        "aspect_ratio_13": aspect_ratio_13,
        "aspect_ratio_23": aspect_ratio_23,
        "radius_mean": radius.mean(),
        "radius_std": radius.std(),
        "radius_cv": radius.std() / (radius.mean() + 1e-8),
        "principal_axes": axes,
    }

def add_side_indentation(points, indent_center=(0.0, 0.5, 0.0), sigma=0.35, depth=0.35):
    """
    Push points inward near a side location to create an asymmetric neck/indentation.
    """
    pts = points.copy()
    c = np.array(indent_center)

    d2 = np.sum((pts - c[None, :]) ** 2, axis=1)
    w = np.exp(-d2 / (2 * sigma ** 2))

    radial = normalize_rows(pts)
    pts = pts - depth * w[:, None] * radial
    return pts

def add_ring_indentation(
    points,
    axis=(1.0, 0.0, 0.0),
    z0=0.0,
    sigma_z=0.20,
    depth=0.20,
    center=None,
    offset=(0.0, 0.0, 0.0)
):
    """
    Create a ring-like constriction around a given axis.

    Parameters
    ----------
    points : (N, 3) array
        Input point cloud.
    axis : tuple or array-like of length 3
        Main axis of the nucleus. The constriction ring is centered
        around a plane orthogonal to this axis.
    z0 : float
        Axial location of the constriction center along the axis.
    sigma_z : float
        Width of the constriction band along the axis.
    depth : float
        Strength of inward contraction.
    center : None or (3,) array
        Optional center of the nucleus. If None, use point cloud centroid.

    Returns
    -------
    pts_new : (N, 3) array
        Deformed point cloud with ring-like inward constriction.
    """
    pts = points.copy()

    if center is None:
        center = pts.mean(axis=0)
    center = np.asarray(center)

    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-8)

    offset = np.asarray(offset, dtype=float)
    offset = offset - np.dot(offset, axis) * axis

    axis_origin = center + offset

    # Coordinates relative to shifted axis
    x = pts - axis_origin[None, :]

    # Axial coordinate along the main axis
    axial = x @ axis

    # Projection to shifted axis line
    proj = np.outer(axial, axis)

    # Radial vector relative to shifted axis
    radial = x - proj
    radial_dir = normalize_rows(radial)

    # Ring band weight along the axis
    w = np.exp(-((axial - z0) ** 2) / (2 * sigma_z ** 2))

    # Move inward toward the shifted axis
    x_new = x - depth * w[:, None] * radial_dir

    pts_new = x_new + axis_origin[None, :]
    return pts_new


def bend_point_cloud_to_arc(
    points,
    axis=(1.0, 0.0, 0.0),
    bend_direction=(0.0, 1.0, 0.0),
    bend_angle=np.pi,
    arc_length_scale=1.0,
    center=None,
):
    """
    Bend a point cloud along `axis` into an arc in the bend plane.

    This is useful for turning a prolate ellipsoid/egg into a crescent-like
    nucleus while preserving the local cross-section coordinates.
    """
    pts = np.asarray(points, dtype=float)
    if center is None:
        center = pts.mean(axis=0)
    center = np.asarray(center, dtype=float)

    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-8)

    bend_direction = np.asarray(bend_direction, dtype=float)
    bend_direction = bend_direction - np.dot(bend_direction, axis) * axis
    bend_direction_norm = np.linalg.norm(bend_direction)
    if bend_direction_norm < 1e-8:
        raise ValueError("bend_direction must not be parallel to axis.")
    bend_direction = bend_direction / bend_direction_norm

    thickness_axis = np.cross(axis, bend_direction)
    thickness_axis_norm = np.linalg.norm(thickness_axis)
    if thickness_axis_norm < 1e-8:
        raise ValueError("axis and bend_direction must define a bend plane.")
    thickness_axis = thickness_axis / thickness_axis_norm

    x = pts - center[None, :]
    axial = x @ axis
    lateral = x @ bend_direction
    thickness = x @ thickness_axis

    axial_min = axial.min()
    axial_max = axial.max()
    axial_span = axial_max - axial_min
    if axial_span < 1e-8 or abs(bend_angle) < 1e-8:
        return pts.copy()

    axial_mid = 0.5 * (axial_min + axial_max)
    theta = ((axial - axial_mid) / (axial_span + 1e-8)) * bend_angle
    radius = arc_length_scale * axial_span / (abs(bend_angle) + 1e-8)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    centerline = (
        radius * sin_theta[:, None] * axis[None, :]
        + radius * (cos_theta - 1.0)[:, None] * bend_direction[None, :]
    )
    radial_dir = (
        sin_theta[:, None] * axis[None, :]
        + cos_theta[:, None] * bend_direction[None, :]
    )

    pts_new = (
        centerline
        + lateral[:, None] * radial_dir
        + thickness[:, None] * thickness_axis[None, :]
    )
    pts_new = pts_new + center[None, :]
    pts_new = pts_new - pts_new.mean(axis=0, keepdims=True) + center[None, :]
    return pts_new


def sample_base_nucleus_point_cloud(
    n=2000,
    axis=(1.0, 0.0, 0.0),
    random_orientation=True,
    size_range=(0.85, 1.20),
    aspect_ratio_2_range=(0.78, 0.96),
    aspect_ratio_3_range=(0.65, 0.90),
    eggness_range=(0.0, 0.28),
    noise_range=(0.005, 0.02),
    random_state=0,
):
    """
    Sample a base normal nucleus from the ellipsoid-to-egg family.

    `eggness=0` reduces to an ellipsoid. Positive `eggness` yields a smooth
    egg-like asymmetry along `axis`.
    """
    rng = np.random.default_rng(random_state)

    size = rng.uniform(*size_range)
    axis_1 = size * rng.uniform(0.95, 1.15)
    axis_2 = axis_1 * rng.uniform(*aspect_ratio_2_range)
    axis_3 = axis_1 * rng.uniform(*aspect_ratio_3_range)
    axes = np.array([axis_1, axis_2, axis_3], dtype=float)

    eggness = rng.uniform(*eggness_range)
    noise = rng.uniform(*noise_range)
    if random_orientation:
        basis = np.stack(random_orthonormal_basis(rng), axis=0)
    else:
        basis = np.stack(orthonormal_basis_from_axis(axis), axis=0)

    pts = sample_egg_shape_point_cloud(
        n=n,
        axes=axes,
        basis=basis,
        eggness=eggness,
        noise=noise,
        random_state=int(rng.integers(0, 1_000_000_000)),
    )

    metadata = {
        "axes": axes,
        "size": float(size),
        "eggness": float(eggness),
        "noise": float(noise),
        "axis": basis[0],
        "basis": basis,
        "deformation": "none",
    }
    return pts, metadata


def sample_ring_indented_nucleus_point_cloud(
    n=2000,
    axis=(1.0, 0.0, 0.0),
    random_orientation=True,
    deformation_axis_index=None,
    base_kwargs=None,
    z0_range=(-0.12, 0.12),
    sigma_z_range=(0.12, 0.22),
    depth_range=(0.25, 0.45),
    offset_2_range=(-0.15, 0.15),
    offset_3_range=(-0.10, 0.10),
    random_state=0,
):
    """Sample an egg/ellipsoid nucleus with a near-centered ring indentation.

    By default the constriction axis is the nucleus long axis.
    """
    rng = np.random.default_rng(random_state)
    base_kwargs = {} if base_kwargs is None else dict(base_kwargs)
    base_kwargs.setdefault("axis", axis)
    base_kwargs.setdefault("random_orientation", random_orientation)
    pts, metadata = sample_base_nucleus_point_cloud(
        n=n,
        **base_kwargs,
        random_state=int(rng.integers(0, 1_000_000_000)),
    )

    axes = metadata["axes"]
    basis = np.asarray(metadata["basis"], dtype=float)
    if deformation_axis_index is None:
        deformation_axis_index = int(np.argmax(axes))
    orth_idx = [i for i in range(3) if i != deformation_axis_index]
    axis = basis[deformation_axis_index]
    axis_2 = basis[orth_idx[0]]
    axis_3 = basis[orth_idx[1]]

    z0 = rng.uniform(*z0_range) * axes[deformation_axis_index]
    sigma_z = rng.uniform(*sigma_z_range) * axes[deformation_axis_index]
    depth = rng.uniform(*depth_range) * min(axes[orth_idx[0]], axes[orth_idx[1]])
    offset_2 = rng.uniform(*offset_2_range) * axes[orth_idx[0]]
    offset_3 = rng.uniform(*offset_3_range) * axes[orth_idx[1]]
    offset = offset_2 * axis_2 + offset_3 * axis_3

    pts = add_ring_indentation(
        pts,
        axis=axis,
        z0=z0,
        sigma_z=sigma_z,
        depth=depth,
        offset=offset,
    )

    metadata.update(
        {
            "deformation": "ring",
            "z0": float(z0),
            "sigma_z": float(sigma_z),
            "depth": float(depth),
            "offset": offset,
            "deformation_axis_index": int(deformation_axis_index),
            "deformation_axis": axis,
        }
    )
    return pts, metadata


def sample_side_indented_nucleus_point_cloud(
    n=2000,
    axis=(1.0, 0.0, 0.0),
    random_orientation=True,
    deformation_axis_index=None,
    base_kwargs=None,
    axial_shift_range=(-0.12, 0.12),
    lateral_position_range=(0.78, 0.98),
    lateral_offset_3_range=(-0.12, 0.12),
    sigma_range=(0.14, 0.25),
    depth_range=(0.22, 0.40),
    random_state=0,
):
    """Sample an egg/ellipsoid nucleus with a near-centered side indentation.

    By default the indentation is positioned relative to the nucleus long axis.
    """
    rng = np.random.default_rng(random_state)
    base_kwargs = {} if base_kwargs is None else dict(base_kwargs)
    base_kwargs.setdefault("axis", axis)
    base_kwargs.setdefault("random_orientation", random_orientation)
    pts, metadata = sample_base_nucleus_point_cloud(
        n=n,
        **base_kwargs,
        random_state=int(rng.integers(0, 1_000_000_000)),
    )

    axes = metadata["axes"]
    basis = np.asarray(metadata["basis"], dtype=float)
    if deformation_axis_index is None:
        deformation_axis_index = int(np.argmax(axes))
    orth_idx = [i for i in range(3) if i != deformation_axis_index]
    axis = basis[deformation_axis_index]

    side_dir_pos = int(rng.integers(0, 2))
    side_dir_index = orth_idx[side_dir_pos]
    other_dir_index = orth_idx[1 - side_dir_pos]
    axis_2 = basis[side_dir_index]
    axis_3 = basis[other_dir_index]
    side_sign = rng.choice([-1.0, 1.0])

    axial_shift = rng.uniform(*axial_shift_range) * axes[deformation_axis_index]
    lateral_2 = side_sign * rng.uniform(*lateral_position_range) * axes[side_dir_index]
    lateral_3 = rng.uniform(*lateral_offset_3_range) * axes[other_dir_index]
    indent_center = (
        axial_shift * axis
        + lateral_2 * axis_2
        + lateral_3 * axis_3
    )

    sigma = rng.uniform(*sigma_range) * np.mean(axes[orth_idx])
    depth = rng.uniform(*depth_range) * min(axes[orth_idx[0]], axes[orth_idx[1]])

    pts = add_side_indentation(
        pts,
        indent_center=indent_center,
        sigma=sigma,
        depth=depth,
    )

    metadata.update(
        {
            "deformation": "side",
            "indent_center": indent_center,
            "sigma": float(sigma),
            "depth": float(depth),
            "deformation_axis_index": int(deformation_axis_index),
            "deformation_axis": axis,
            "side_direction_index": int(side_dir_index),
        }
    )
    return pts, metadata


def sample_c_shape_nucleus_point_cloud(
    n=2000,
    axis=(1.0, 0.0, 0.0),
    random_orientation=True,
    deformation_axis_index=None,
    bend_direction_index=None,
    base_kwargs=None,
    bend_angle_range=(1.10 * np.pi, 1.65 * np.pi),
    arc_length_scale_range=(1.00, 1.15),
    random_state=0,
):
    """Sample a C-shaped nucleus by bending an elongated egg/ellipsoid."""
    rng = np.random.default_rng(random_state)
    base_kwargs = {} if base_kwargs is None else dict(base_kwargs)
    base_kwargs.setdefault("axis", axis)
    base_kwargs.setdefault("random_orientation", random_orientation)
    base_kwargs.setdefault("size_range", (0.95, 1.25))
    base_kwargs.setdefault("aspect_ratio_2_range", (0.28, 0.48))
    base_kwargs.setdefault("aspect_ratio_3_range", (0.22, 0.42))
    base_kwargs.setdefault("eggness_range", (0.0, 0.18))
    base_kwargs.setdefault("noise_range", (0.004, 0.015))

    pts, metadata = sample_base_nucleus_point_cloud(
        n=n,
        **base_kwargs,
        random_state=int(rng.integers(0, 1_000_000_000)),
    )

    axes = metadata["axes"]
    basis = np.asarray(metadata["basis"], dtype=float)
    if deformation_axis_index is None:
        deformation_axis_index = int(np.argmax(axes))

    orth_idx = [i for i in range(3) if i != deformation_axis_index]
    if bend_direction_index is None:
        bend_direction_index = int(orth_idx[int(rng.integers(0, 2))])
    if bend_direction_index not in orth_idx:
        raise ValueError("bend_direction_index must be one of the transverse axis indices.")

    bend_sign = rng.choice([-1.0, 1.0])
    bend_angle = rng.uniform(*bend_angle_range)
    arc_length_scale = rng.uniform(*arc_length_scale_range)
    deformation_axis = basis[deformation_axis_index]
    bend_direction = bend_sign * basis[bend_direction_index]

    pts = bend_point_cloud_to_arc(
        pts,
        axis=deformation_axis,
        bend_direction=bend_direction,
        bend_angle=bend_angle,
        arc_length_scale=arc_length_scale,
    )

    metadata.update(
        {
            "deformation": "c_shape",
            "bend_angle": float(bend_angle),
            "bend_angle_degrees": float(np.degrees(bend_angle)),
            "arc_length_scale": float(arc_length_scale),
            "deformation_axis_index": int(deformation_axis_index),
            "deformation_axis": deformation_axis,
            "bend_direction_index": int(bend_direction_index),
            "bend_direction": bend_direction,
        }
    )
    return pts, metadata


def build_simulated_nucleus_dataset(
    n_normal=450,
    n_ring=35,
    n_side=15,
    n_c_shape=0,
    points_per_nucleus=2000,
    axis=(1.0, 0.0, 0.0),
    random_orientation=True,
    base_kwargs=None,
    ring_kwargs=None,
    side_kwargs=None,
    c_shape_kwargs=None,
    random_state=0,
):
    """
    Build a simulated dataset of normal and deformed nuclei.

    Returns
    -------
    dataset : dict
        Keys:
        - `points`: (N, P, 3) array of point clouds
        - `labels`: (N,) array, 0 normal, 1 ring, 2 side, 3 c_shape
        - `subtypes`: (N,) array with values `normal`, `ring`, `side`, `c_shape`
        - `metadata`: list of per-sample dictionaries
    """
    rng = np.random.default_rng(random_state)
    base_kwargs = {} if base_kwargs is None else dict(base_kwargs)
    base_kwargs.setdefault("axis", axis)
    base_kwargs.setdefault("random_orientation", random_orientation)
    ring_kwargs = {} if ring_kwargs is None else dict(ring_kwargs)
    side_kwargs = {} if side_kwargs is None else dict(side_kwargs)
    c_shape_kwargs = {} if c_shape_kwargs is None else dict(c_shape_kwargs)

    point_clouds = []
    labels = []
    subtypes = []
    metadata = []

    for _ in range(n_normal):
        pts, meta = sample_base_nucleus_point_cloud(
            n=points_per_nucleus,
            **base_kwargs,
            random_state=int(rng.integers(0, 1_000_000_000)),
        )
        meta.update({"label": 0, "subtype": "normal"})
        point_clouds.append(pts)
        labels.append(0)
        subtypes.append("normal")
        metadata.append(meta)

    for _ in range(n_ring):
        pts, meta = sample_ring_indented_nucleus_point_cloud(
            n=points_per_nucleus,
            axis=axis,
            random_orientation=random_orientation,
            base_kwargs=base_kwargs,
            **ring_kwargs,
            random_state=int(rng.integers(0, 1_000_000_000)),
        )
        meta.update({"label": 1, "subtype": "ring"})
        point_clouds.append(pts)
        labels.append(1)
        subtypes.append("ring")
        metadata.append(meta)

    for _ in range(n_side):
        pts, meta = sample_side_indented_nucleus_point_cloud(
            n=points_per_nucleus,
            axis=axis,
            random_orientation=random_orientation,
            base_kwargs=base_kwargs,
            **side_kwargs,
            random_state=int(rng.integers(0, 1_000_000_000)),
        )
        meta.update({"label": 2, "subtype": "side"})
        point_clouds.append(pts)
        labels.append(2)
        subtypes.append("side")
        metadata.append(meta)

    for _ in range(n_c_shape):
        pts, meta = sample_c_shape_nucleus_point_cloud(
            n=points_per_nucleus,
            axis=axis,
            random_orientation=random_orientation,
            base_kwargs=base_kwargs,
            **c_shape_kwargs,
            random_state=int(rng.integers(0, 1_000_000_000)),
        )
        meta.update({"label": 3, "subtype": "c_shape"})
        point_clouds.append(pts)
        labels.append(3)
        subtypes.append("c_shape")
        metadata.append(meta)

    order = rng.permutation(len(point_clouds))
    point_clouds = np.stack([point_clouds[i] for i in order], axis=0)
    labels = np.asarray([labels[i] for i in order], dtype=int)
    subtypes = np.asarray([subtypes[i] for i in order], dtype=object)
    metadata = [metadata[i] for i in order]

    return {
        "points": point_clouds,
        "labels": labels,
        "subtypes": subtypes,
        "metadata": metadata,
    }


def plot_simulated_dataset_gallery(
    dataset,
    samples_per_subtype=10,
    subtypes=("normal", "ring", "side",'c_shape'),
    figsize=(24, 8),
    point_size=1.0,
    alpha=0.75,
    elev=18,
    azim=35,
    cmap="viridis",
    show=True,
    return_fig=False,
    random_state=0,
    title=None,
    points_key="points",
):
    """
    Plot a static matplotlib gallery of simulated nuclei.

    Rows correspond to subtypes and columns to randomly sampled nuclei within
    each subtype. If given, `title` is shown as the figure-level title.
    """
    points = np.asarray(dataset[points_key])
    dataset_subtypes = np.asarray(dataset["subtypes"])
    chosen_indices = _select_gallery_indices(
        dataset_subtypes=dataset_subtypes,
        samples_per_subtype=samples_per_subtype,
        subtypes=subtypes,
        random_state=random_state,
    )
    point_cloud_rows = [[points[idx] for idx in row] for row in chosen_indices]

    return _plot_static_point_cloud_gallery(
        point_cloud_rows=point_cloud_rows,
        row_labels=subtypes,
        ncols=samples_per_subtype,
        figsize=figsize,
        point_size=point_size,
        alpha=alpha,
        elev=elev,
        azim=azim,
        cmap=cmap,
        show=show,
        return_fig=return_fig,
        title=title,
    )


def _select_gallery_indices(dataset_subtypes, samples_per_subtype, subtypes, random_state):
    rng = np.random.default_rng(random_state)
    chosen_indices = []

    for subtype in subtypes:
        subtype_idx = np.flatnonzero(dataset_subtypes == subtype)
        if len(subtype_idx) == 0:
            chosen_indices.append(np.array([], dtype=int))
            continue

        take = min(samples_per_subtype, len(subtype_idx))
        chosen = rng.choice(subtype_idx, size=take, replace=False)
        chosen_indices.append(chosen)

    return chosen_indices


def _compute_gallery_limits(point_clouds):
    all_pts = np.concatenate(point_clouds, axis=0)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    half_range = 0.55 * np.max(maxs - mins)
    return center, half_range


def _plot_static_point_cloud_gallery(
    point_cloud_rows,
    row_labels,
    ncols,
    figsize,
    point_size,
    alpha,
    elev,
    azim,
    cmap,
    show,
    return_fig,
    limits_reference=None,
    title=None,
):
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    import matplotlib.pyplot as plt

    selected = [pts for row in point_cloud_rows for pts in row]
    if not selected:
        raise ValueError("No matching point clouds found for the requested subtypes.")

    reference_clouds = selected if limits_reference is None else limits_reference
    center, half_range = _compute_gallery_limits(reference_clouds)

    fig = plt.figure(figsize=figsize)
    if title is not None:
        fig.suptitle(title, fontsize=22, y=0.98)
    nrows = len(row_labels)

    plot_idx = 1
    for row_idx, row_label in enumerate(row_labels):
        row_points = point_cloud_rows[row_idx]
        for col_idx in range(ncols):
            ax = fig.add_subplot(nrows, ncols, plot_idx, projection="3d")
            plot_idx += 1

            if col_idx >= len(row_points):
                ax.axis("off")
                continue

            pts = row_points[col_idx]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                c=pts[:, 2],
                cmap=cmap,
                s=point_size,
                alpha=alpha,
                linewidths=0,
            )
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlim(center[0] - half_range, center[0] + half_range)
            ax.set_ylim(center[1] - half_range, center[1] + half_range)
            ax.set_zlim(center[2] - half_range, center[2] + half_range)
            ax.set_box_aspect((1, 1, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.grid(False)

            if col_idx == 0:
                ax.set_title(row_label, fontsize=20, pad=8, loc="left")

    if title is not None:
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        fig.tight_layout()
    if show:
        plt.show()
    if return_fig:
        return fig
    plt.close(fig)
    return None


def plot_reconstructed_dataset_gallery(
    model,
    dataset,
    device=None,
    samples_per_subtype=10,
    subtypes=("normal", "ring", "side",'c_shape'),
    figsize=(24, 8),
    point_size=1.0,
    alpha=0.75,
    elev=18,
    azim=35,
    cmap="viridis",
    show=True,
    return_fig=False,
    random_state=0,
    title=None,
    points_key="points",
):
    """
    Plot PointNet-style reconstructions using the same gallery layout as the
    simulated dataset gallery.

    The same random subtype sampling is reused, so matching `random_state`
    will give the same 30 positions as `plot_simulated_dataset_gallery`.
    If given, `title` is shown as the figure-level title.
    """
    points = np.asarray(dataset[points_key])
    dataset_subtypes = np.asarray(dataset["subtypes"])
    chosen_indices = _select_gallery_indices(
        dataset_subtypes=dataset_subtypes,
        samples_per_subtype=samples_per_subtype,
        subtypes=subtypes,
        random_state=random_state,
    )

    input_rows = [[points[idx] for idx in row] for row in chosen_indices]
    selected_inputs = [pts for row in input_rows for pts in row]
    if not selected_inputs:
        raise ValueError("No matching point clouds found for the requested subtypes.")

    reconstruct_result = model.reconstruct(np.stack(selected_inputs, axis=0), device=device)
    reconstructed = reconstruct_result[0] if isinstance(reconstruct_result, tuple) else reconstruct_result
    reconstructed = np.asarray(reconstructed)
    if reconstructed.ndim == 2:
        reconstructed = reconstructed[None, ...]

    point_cloud_rows = []
    start = 0
    for row in input_rows:
        stop = start + len(row)
        point_cloud_rows.append([reconstructed[idx] for idx in range(start, stop)])
        start = stop

    return _plot_static_point_cloud_gallery(
        point_cloud_rows=point_cloud_rows,
        row_labels=subtypes,
        ncols=samples_per_subtype,
        figsize=figsize,
        point_size=point_size,
        alpha=alpha,
        elev=elev,
        azim=azim,
        cmap=cmap,
        show=show,
        return_fig=return_fig,
        limits_reference=selected_inputs,
        title=title,
    )


def plot_point_cloud(
    points,
    title="3D Point Cloud",
    color=None,
    size=2,
    highlight_q=None,
    important_color="red",
    other_color="black",
    other_opacity=0.4,
    important_opacity=0.9,
):
    """Interactive 3D plot with Plotly."""
    pts = points.copy()
    if color is None:
        color = pts[:, 2]

    use_importance = (
        highlight_q is not None
        and np.ndim(color) == 1
        and len(color) == len(pts)
    )

    if use_importance:
        scores = np.asarray(color)
        cutoff = np.quantile(scores, highlight_q)
        mask = scores >= cutoff

        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=pts[~mask, 0],
                y=pts[~mask, 1],
                z=pts[~mask, 2],
                mode="markers",
                marker=dict(
                    size=max(size * 0.8, 1.0),
                    color=other_color,
                    opacity=other_opacity,
                ),
                name="Other",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=pts[mask, 0],
                y=pts[mask, 1],
                z=pts[mask, 2],
                mode="markers",
                marker=dict(
                    size=max(size * 1.2, 1.0),
                    color=important_color,
                    opacity=important_opacity,
                ),
                name="Important",
            )
        )
    else:
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="markers",
                    marker=dict(
                        size=size,
                        color=color,
                        opacity=0.8,
                        colorscale="Viridis"
                    )
                )
            ]
        )

    fig.update_layout(
        title=title,
        width=500,
        height=500,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.show()
