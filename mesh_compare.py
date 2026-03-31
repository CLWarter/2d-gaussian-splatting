import argparse
import csv
import itertools
import os
import re
from pathlib import Path

import numpy as np
import open3d as o3d


OURS_RE = re.compile(r"^ours_(\d+)$")


def load_mesh(path: str) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty():
        raise ValueError(f"Mesh is empty or could not be loaded: {path}")
    if not mesh.has_triangles():
        raise ValueError(f"Mesh has no triangles: {path}")
    mesh.compute_vertex_normals()
    return mesh


def sample_points(mesh: o3d.geometry.TriangleMesh, n_points: int) -> o3d.geometry.PointCloud:
    return mesh.sample_points_uniformly(number_of_points=n_points)


def compute_bidirectional_distances(
    pcd_a: o3d.geometry.PointCloud,
    pcd_b: o3d.geometry.PointCloud
):
    dists_a_to_b = np.asarray(pcd_a.compute_point_cloud_distance(pcd_b), dtype=np.float64)
    dists_b_to_a = np.asarray(pcd_b.compute_point_cloud_distance(pcd_a), dtype=np.float64)
    return dists_a_to_b, dists_b_to_a


def chamfer_distance(dists_a_to_b: np.ndarray, dists_b_to_a: np.ndarray) -> float:
    return float(np.mean(dists_a_to_b) + np.mean(dists_b_to_a))


def hausdorff_distance(dists_a_to_b: np.ndarray, dists_b_to_a: np.ndarray) -> float:
    return float(max(np.max(dists_a_to_b), np.max(dists_b_to_a)))


def fscore(dists_a_to_b: np.ndarray, dists_b_to_a: np.ndarray, threshold: float) -> float:
    precision = float(np.mean(dists_a_to_b < threshold))
    recall = float(np.mean(dists_b_to_a < threshold))
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def distances_to_colors(dists: np.ndarray, clip_max: float | None = None) -> np.ndarray:
    vals = dists.copy()
    if clip_max is None:
        clip_max = np.percentile(vals, 95.0)
    clip_max = max(float(clip_max), 1e-12)
    vals = np.clip(vals / clip_max, 0.0, 1.0)

    colors = np.zeros((len(vals), 3), dtype=np.float64)

    for i, v in enumerate(vals):
        if v < 0.25:
            t = v / 0.25
            colors[i] = [0.0, t, 1.0]           # blue -> cyan
        elif v < 0.5:
            t = (v - 0.25) / 0.25
            colors[i] = [0.0, 1.0, 1.0 - t]     # cyan -> green
        elif v < 0.75:
            t = (v - 0.5) / 0.25
            colors[i] = [t, 1.0, 0.0]           # green -> yellow
        else:
            t = (v - 0.75) / 0.25
            colors[i] = [1.0, 1.0 - t, 0.0]     # yellow -> red

    return colors


def save_colored_distance_pcd(
    source_pcd: o3d.geometry.PointCloud,
    dists: np.ndarray,
    out_path: str,
    clip_max: float | None = None
):
    pcd = o3d.geometry.PointCloud(source_pcd)
    pcd.colors = o3d.utility.Vector3dVector(distances_to_colors(dists, clip_max=clip_max))
    ok = o3d.io.write_point_cloud(out_path, pcd)
    if not ok:
        raise IOError(f"Failed to write point cloud: {out_path}")


def find_highest_export_mesh(output_root: Path, mesh_name: str) -> dict[str, str] | None:
    train_dir = output_root / "train"
    if not train_dir.is_dir():
        return None

    best_iter = None
    best_dir = None

    for child in train_dir.iterdir():
        if not child.is_dir():
            continue
        m = OURS_RE.match(child.name)
        if not m:
            continue
        it = int(m.group(1))
        mesh_path = child / mesh_name
        if mesh_path.is_file():
            if best_iter is None or it > best_iter:
                best_iter = it
                best_dir = child

    if best_iter is None or best_dir is None:
        return None

    return {
        "name": output_root.name,
        "iteration": str(best_iter),
        "mesh_path": str(best_dir / mesh_name),
    }


def find_meshes_in_outputs(base_dir: str, mesh_name: str) -> list[dict[str, str]]:
    base = Path(base_dir)
    if not base.is_dir():
        raise ValueError(f"Base directory does not exist: {base_dir}")

    found = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        result = find_highest_export_mesh(child, mesh_name)
        if result is not None:
            found.append(result)

    return found


def print_bbox_info(meshes: dict[str, o3d.geometry.TriangleMesh]):
    print("\nBounding box extents:")
    for name, mesh in meshes.items():
        bbox = mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        print(f"  {name}: extent = {extent}")
    print("\nMake sure all meshes are aligned and in the same coordinate system.\n")


def main():
    parser = argparse.ArgumentParser(description="Find meshes in ./output and compare them pairwise.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./output",
        help="Base directory containing output runs"
    )
    parser.add_argument(
        "--mesh_name",
        type=str,
        default="fuse.ply",
        help="Mesh filename inside train/ours_ITER/"
    )
    parser.add_argument(
        "--sample_points",
        type=int,
        default=100000,
        help="Number of points sampled per mesh"
    )
    parser.add_argument(
        "--fscore_threshold",
        type=float,
        default=0.01,
        help="Distance threshold for F-score"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="mesh_compare_out",
        help="Output directory for CSV and colored point clouds"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open Open3D viewer for saved colored point clouds"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Searching for meshes in: {args.base_dir}")
    found_meshes = find_meshes_in_outputs(args.base_dir, args.mesh_name)

    if not found_meshes:
        print("No meshes found.")
        return

    print("\nFound meshes:")
    for item in found_meshes:
        print(f"  {item['name']}: iter {item['iteration']} -> {item['mesh_path']}")

    print("\nLoading meshes...")
    meshes: dict[str, o3d.geometry.TriangleMesh] = {}
    sampled_pcds: dict[str, o3d.geometry.PointCloud] = {}

    for item in found_meshes:
        run_name = item["name"]
        iteration = item["iteration"]
        label = f"{run_name}_iter{iteration}"
        mesh_path = item["mesh_path"]

        mesh = load_mesh(mesh_path)
        meshes[label] = mesh
        print(f"  loaded: {label}")

    print_bbox_info(meshes)

    print("Sampling point clouds...")
    for label, mesh in meshes.items():
        sampled_pcds[label] = sample_points(mesh, args.sample_points)
        print(f"  sampled: {label}")

    pairs = list(itertools.combinations(sampled_pcds.keys(), 2))
    if not pairs:
        print("Need at least two meshes to compare.")
        return

    csv_path = os.path.join(args.out_dir, "pairwise_metrics.csv")
    rows = []

    print(f"\nComparing {len(pairs)} pairs...\n")
    for name_a, name_b in pairs:
        print(f"Comparing {name_a} <-> {name_b}")

        pcd_a = sampled_pcds[name_a]
        pcd_b = sampled_pcds[name_b]

        dists_a_to_b, dists_b_to_a = compute_bidirectional_distances(pcd_a, pcd_b)

        chamfer = chamfer_distance(dists_a_to_b, dists_b_to_a)
        hausdorff = hausdorff_distance(dists_a_to_b, dists_b_to_a)
        f1 = fscore(dists_a_to_b, dists_b_to_a, args.fscore_threshold)

        mean_a_to_b = float(np.mean(dists_a_to_b))
        mean_b_to_a = float(np.mean(dists_b_to_a))
        median_a_to_b = float(np.median(dists_a_to_b))
        median_b_to_a = float(np.median(dists_b_to_a))
        max_a_to_b = float(np.max(dists_a_to_b))
        max_b_to_a = float(np.max(dists_b_to_a))

        print(f"  Chamfer   : {chamfer:.8f}")
        print(f"  Hausdorff : {hausdorff:.8f}")
        print(f"  F-score   : {f1:.8f} (threshold={args.fscore_threshold})")

        rows.append({
            "mesh_a": name_a,
            "mesh_b": name_b,
            "mean_a_to_b": mean_a_to_b,
            "mean_b_to_a": mean_b_to_a,
            "median_a_to_b": median_a_to_b,
            "median_b_to_a": median_b_to_a,
            "max_a_to_b": max_a_to_b,
            "max_b_to_a": max_b_to_a,
            "chamfer": chamfer,
            "hausdorff": hausdorff,
            "fscore": f1,
            "threshold": args.fscore_threshold,
        })

        clip_val = max(
            np.percentile(dists_a_to_b, 95.0),
            np.percentile(dists_b_to_a, 95.0)
        )

        out_a = os.path.join(args.out_dir, f"{name_a}_to_{name_b}_dist.ply")
        out_b = os.path.join(args.out_dir, f"{name_b}_to_{name_a}_dist.ply")

        save_colored_distance_pcd(pcd_a, dists_a_to_b, out_a, clip_max=clip_val)
        save_colored_distance_pcd(pcd_b, dists_b_to_a, out_b, clip_max=clip_val)

        print(f"  saved: {out_a}")
        print(f"  saved: {out_b}")

        if args.visualize:
            vis_a = o3d.io.read_point_cloud(out_a)
            vis_b = o3d.io.read_point_cloud(out_b)
            o3d.visualization.draw_geometries([vis_a], window_name=f"{name_a} -> {name_b}")
            o3d.visualization.draw_geometries([vis_b], window_name=f"{name_b} -> {name_a}")

        print()

    fieldnames = [
        "mesh_a", "mesh_b",
        "mean_a_to_b", "mean_b_to_a",
        "median_a_to_b", "median_b_to_a",
        "max_a_to_b", "max_b_to_a",
        "chamfer", "hausdorff",
        "fscore", "threshold"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved metrics CSV: {csv_path}")

    rows_sorted = sorted(rows, key=lambda r: r["chamfer"])
    print("\nPairs sorted by Chamfer (lower = more similar):")
    for r in rows_sorted:
        print(
            f"  {r['mesh_a']} <-> {r['mesh_b']}: "
            f"Chamfer={r['chamfer']:.8f}, "
            f"Hausdorff={r['hausdorff']:.8f}, "
            f"F-score={r['fscore']:.8f}"
        )


if __name__ == "__main__":
    main()