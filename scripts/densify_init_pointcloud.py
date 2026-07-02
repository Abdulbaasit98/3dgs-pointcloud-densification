# densify_init_pointcloud.py
#
# Neighbor-interpolated densification of a COLMAP sparse point cloud, used to
# give 3D Gaussian Splatting a denser, better-distributed initialization than
# the raw SfM output.
#
# Why: 3DGS initializes one Gaussian per point in the sparse SfM point cloud,
# then relies on Adaptive Density Control (densify/split/clone during
# training) to fill in gaps. Under-sampled regions of the SfM point cloud
# mean those areas start with too few Gaussians and take longer to converge.
# This script inserts extra points at the midpoints of any neighbor pair
# whose distance exceeds a threshold (i.e. "gaps" in the point cloud),
# interpolating color from the two endpoints. It is a purely geometric,
# CPU-only preprocessing step -- it does not touch the 3DGS training code at
# all. You just point training at the densified .ply instead of the original.
#
# Accepts EITHER a COLMAP points3D.bin (as shipped by the official
# pre-packaged datasets) OR a points3D.ply -- auto-detected from the file
# extension. Always writes a .ply, since that's what the 3DGS repo's
# fetchPly() reads.
#
# Runs entirely on CPU. No GPU, no PyTorch required for this step -- develop
# and test this on your laptop before ever touching Colab.
#
# Dependencies: numpy, scipy, plyfile
#   pip install numpy scipy plyfile

import argparse
import os
import struct
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement


def load_points3D_bin(path):
    """
    Minimal reader for COLMAP's binary points3D.bin format (matches the
    format used by the 3DGS repo's own scene/colmap_loader.py:
    read_points3D_binary). Only reads xyz + rgb; ignores track/error info,
    which 3DGS's own initialization does not use either.
    """
    xyz = []
    rgb = []
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            # point3D_id (Q), xyz (3d), rgb (3B), error (d)
            f.read(8)  # point3D_id
            x, y, z = struct.unpack("<ddd", f.read(24))
            r, g, b = struct.unpack("<BBB", f.read(3))
            f.read(8)  # error
            track_length = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * track_length)  # track elements (image_id, point2D_idx), each 4+4 bytes
            xyz.append((x, y, z))
            rgb.append((r, g, b))
    return np.array(xyz, dtype=np.float32), np.array(rgb, dtype=np.uint8)


def load_ply(path):
    """Load xyz positions and RGB colors from a COLMAP-style points3D.ply,
    matching the schema written by the 3DGS repo's storePly()."""
    ply = PlyData.read(path)
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.uint8)
    return xyz, rgb


def load_points3D(path):
    """Dispatch to the right reader based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bin":
        return load_points3D_bin(path)
    elif ext == ".ply":
        return load_ply(path)
    else:
        raise ValueError(f"Unsupported input extension '{ext}' -- expected .bin or .ply")


def save_ply(path, xyz, rgb):
    """Write xyz/rgb back out in the same schema the 3DGS repo's fetchPly()
    expects (x,y,z, nx,ny,nz, red,green,blue). Normals are zeroed -- the
    repo does not use them for Gaussian initialization."""
    normals = np.zeros_like(xyz)
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attrs = np.concatenate([xyz, normals, rgb.astype(np.float32)], axis=1)
    elements[:] = list(map(tuple, attrs))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el], text=False).write(path)


def densify(xyz, rgb, k=6, factor=1.5, max_new_per_point=3, seed=0):
    """
    Insert new points at the midpoints of under-sampled neighbor pairs.

    Args:
        xyz: [N, 3] point positions.
        rgb: [N, 3] uint8 colors.
        k: number of nearest neighbors considered per point.
        factor: a neighbor pair is considered a "gap" (and gets a new point)
                if its distance exceeds factor * median_nearest_neighbor_distance.
                Higher factor = fewer, more conservative insertions.
        max_new_per_point: cap on new points inserted per original point,
                            to avoid runaway growth in sparse outlier regions.

    Returns:
        (aug_xyz, aug_rgb): original points with new points appended.
    """
    rng = np.random.default_rng(seed)
    tree = cKDTree(xyz)
    dists, idxs = tree.query(xyz, k=k + 1)  # column 0 is the point itself

    median_nn_dist = np.median(dists[:, 1])
    threshold = factor * median_nn_dist
    print(f"Median nearest-neighbor distance: {median_nn_dist:.5f}")
    print(f"Gap threshold ({factor}x median): {threshold:.5f}")

    new_xyz, new_rgb = [], []
    seen_pairs = set()

    for i in range(xyz.shape[0]):
        inserted = 0
        for rank in range(1, k + 1):
            if inserted >= max_new_per_point:
                break
            j = int(idxs[i, rank])
            d = dists[i, rank]
            if d <= threshold:
                continue
            pair = (min(i, j), max(i, j))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            midpoint = (xyz[i] + xyz[j]) / 2.0
            mid_color = ((rgb[i].astype(np.float32) + rgb[j].astype(np.float32)) / 2.0).astype(np.uint8)
            new_xyz.append(midpoint)
            new_rgb.append(mid_color)
            inserted += 1

    if new_xyz:
        aug_xyz = np.concatenate([xyz, np.stack(new_xyz)], axis=0)
        aug_rgb = np.concatenate([rgb, np.stack(new_rgb)], axis=0)
    else:
        aug_xyz, aug_rgb = xyz, rgb

    return aug_xyz, aug_rgb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to original points3D.ply")
    parser.add_argument("--output", required=True, help="Path to write densified .ply")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--factor", type=float, default=1.5)
    parser.add_argument("--max_new_per_point", type=int, default=3)
    args = parser.parse_args()

    xyz, rgb = load_points3D(args.input)
    print(f"Original point count: {xyz.shape[0]}")

    aug_xyz, aug_rgb = densify(xyz, rgb, k=args.k, factor=args.factor,
                                max_new_per_point=args.max_new_per_point)
    print(f"Densified point count: {aug_xyz.shape[0]} (+{aug_xyz.shape[0] - xyz.shape[0]})")

    save_ply(args.output, aug_xyz, aug_rgb)
    print(f"Saved to {args.output}")
