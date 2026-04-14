import argparse
import os
from pathlib import Path

import numpy as np
import open3d as o3d


def load_mesh(path: str) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty():
        raise ValueError(f"Mesh is empty or could not be loaded: {path}")
    if not mesh.has_triangles():
        raise ValueError(f"Mesh has no triangles: {path}")
    mesh.compute_vertex_normals()
    return mesh


def sample_points(
    mesh: o3d.geometry.TriangleMesh,
    n_points: int,
    method: str = "uniform"
) -> o3d.geometry.PointCloud:
    method = method.lower()
    if method == "uniform":
        return mesh.sample_points_uniformly(number_of_points=n_points)
    if method == "poisson":
        return mesh.sample_points_poisson_disk(number_of_points=n_points)
    raise ValueError(f"Unknown sampling method: {method}")


def expanded_bbox_from_mesh(
    mesh: o3d.geometry.TriangleMesh,
    padding: float
) -> o3d.geometry.AxisAlignedBoundingBox:
    bbox = mesh.get_axis_aligned_bounding_box()
    min_b = bbox.get_min_bound() - padding
    max_b = bbox.get_max_bound() + padding
    return o3d.geometry.AxisAlignedBoundingBox(min_b, max_b)


def filter_pointcloud_by_bbox(
    pcd: o3d.geometry.PointCloud,
    bbox: o3d.geometry.AxisAlignedBoundingBox
) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        return o3d.geometry.PointCloud()

    min_b = bbox.get_min_bound()
    max_b = bbox.get_max_bound()

    mask = np.all((pts >= min_b) & (pts <= max_b), axis=1)
    filtered_pts = pts[mask]

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(filtered_pts)
    return out


def compute_distances(
    src_pcd: o3d.geometry.PointCloud,
    dst_pcd: o3d.geometry.PointCloud
) -> np.ndarray:
    return np.asarray(src_pcd.compute_point_cloud_distance(dst_pcd), dtype=np.float64)


def classify_distance_colors(
    dists: np.ndarray,
    good_thresh: float,
    bad_thresh: float
) -> np.ndarray:
    """
    Green  : close / good
    Yellow : medium
    Red    : bad / missing / too much
    """
    colors = np.zeros((len(dists), 3), dtype=np.float64)

    for i, d in enumerate(dists):
        if d <= good_thresh:
            colors[i] = [0.0, 1.0, 0.0]      # green
        elif d <= bad_thresh:
            colors[i] = [1.0, 0.75, 0.0]     # orange/yellow
        else:
            colors[i] = [1.0, 0.0, 0.0]      # red

    return colors


def save_colored_pointcloud(
    pcd: o3d.geometry.PointCloud,
    colors: np.ndarray,
    out_path: str
) -> None:
    out = o3d.geometry.PointCloud(pcd)
    out.colors = o3d.utility.Vector3dVector(colors)

    ok = o3d.io.write_point_cloud(out_path, out)
    if not ok:
        raise IOError(f"Failed to write point cloud: {out_path}")


def print_stats(name: str, dists: np.ndarray) -> None:
    print(f"\n[{name}]")
    print(f"  count  : {len(dists)}")
    print(f"  mean   : {np.mean(dists):.8f}")
    print(f"  median : {np.median(dists):.8f}")
    print(f"  max    : {np.max(dists):.8f}")


def visualize_two(title_a: str, pcd_a_path: str, title_b: str, pcd_b_path: str) -> None:
    pcd_a = o3d.io.read_point_cloud(pcd_a_path)
    pcd_b = o3d.io.read_point_cloud(pcd_b_path)

    o3d.visualization.draw_geometries([pcd_a], window_name=title_a)
    o3d.visualization.draw_geometries([pcd_b], window_name=title_b)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visual mesh-vs-GT comparison.\n"
            "GT->Pred colors where GT is missing.\n"
            "Pred(ROI)->GT colors where prediction has too much/wrong geometry near objects."
        )
    )
    parser.add_argument("--pred_mesh", required=True, type=str, help="Predicted mesh path")
    parser.add_argument("--gt_mesh", required=True, type=str, help="Ground-truth mesh path")
    parser.add_argument("--out_dir", default="visual_mesh_compare_out", type=str)
    parser.add_argument("--sample_points", default=200000, type=int)
    parser.add_argument("--sampling", default="uniform", choices=["uniform", "poisson"])
    parser.add_argument(
        "--roi_padding",
        default=0.2,
        type=float,
        help="Padding around GT bbox for ROI-restricted pred->gt visualization"
    )
    parser.add_argument(
        "--good_thresh",
        default=0.02,
        type=float,
        help="Distance <= this is colored green"
    )
    parser.add_argument(
        "--bad_thresh",
        default=0.05,
        type=float,
        help="Distance > this is colored red; between good and bad is orange"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open Open3D viewer windows after saving files"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[Load] GT   : {args.gt_mesh}", flush=True)
    gt_mesh = load_mesh(args.gt_mesh)

    print(f"[Load] Pred : {args.pred_mesh}", flush=True)
    pred_mesh = load_mesh(args.pred_mesh)

    print(f"[Sample] Sampling {args.sample_points} GT points", flush=True)
    gt_pcd = sample_points(gt_mesh, args.sample_points, method=args.sampling)

    print(f"[Sample] Sampling {args.sample_points} prediction points", flush=True)
    pred_pcd = sample_points(pred_mesh, args.sample_points, method=args.sampling)

    # 1) GT -> Pred  (missing GT parts show up as red on GT)
    print("[Compute] GT -> Pred distances", flush=True)
    gt_to_pred = compute_distances(gt_pcd, pred_pcd)
    print_stats("GT -> Pred (missing-on-pred view)", gt_to_pred)

    gt_colors = classify_distance_colors(
        gt_to_pred,
        good_thresh=args.good_thresh,
        bad_thresh=args.bad_thresh
    )

    gt_missing_out = os.path.join(args.out_dir, "gt_to_pred_missing_view.ply")
    save_colored_pointcloud(gt_pcd, gt_colors, gt_missing_out)
    print(f"[Save] {gt_missing_out}", flush=True)

    # 2) Pred ROI -> GT (too much/wrong local geometry shows up as red on prediction)
    print("[ROI] Filtering prediction points by GT bbox", flush=True)
    gt_bbox = expanded_bbox_from_mesh(gt_mesh, padding=args.roi_padding)
    pred_roi_pcd = filter_pointcloud_by_bbox(pred_pcd, gt_bbox)

    pred_roi_count = len(pred_roi_pcd.points)
    pred_full_count = len(pred_pcd.points)
    print(
        f"[ROI] Kept {pred_roi_count}/{pred_full_count} prediction points "
        f"inside padded GT bbox (padding={args.roi_padding})",
        flush=True
    )

    if pred_roi_count == 0:
        raise ValueError(
            "No predicted sample points remained inside GT ROI. Increase --roi_padding."
        )

    print("[Compute] Pred(ROI) -> GT distances", flush=True)
    pred_roi_to_gt = compute_distances(pred_roi_pcd, gt_pcd)
    print_stats("Pred(ROI) -> GT (too-much/wrong-local-geometry view)", pred_roi_to_gt)

    pred_roi_colors = classify_distance_colors(
        pred_roi_to_gt,
        good_thresh=args.good_thresh,
        bad_thresh=args.bad_thresh
    )

    pred_roi_out = os.path.join(args.out_dir, "pred_roi_to_gt_extra_view.ply")
    save_colored_pointcloud(pred_roi_pcd, pred_roi_colors, pred_roi_out)
    print(f"[Save] {pred_roi_out}", flush=True)

    # Save bbox mesh for optional inspection
    bbox_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(gt_bbox)
    bbox_out = os.path.join(args.out_dir, "gt_roi_bbox.ply")
    o3d.io.write_line_set(bbox_out, bbox_lines)
    print(f"[Save] {bbox_out}", flush=True)

    print("\nInterpretation:", flush=True)
    print("  gt_to_pred_missing_view.ply", flush=True)
    print("    Green  = GT surface is closely reconstructed", flush=True)
    print("    Orange = medium mismatch", flush=True)
    print("    Red    = GT surface is missing / far from prediction", flush=True)

    print("  pred_roi_to_gt_extra_view.ply", flush=True)
    print("    Green  = prediction near objects lies close to GT", flush=True)
    print("    Orange = medium local mismatch", flush=True)
    print("    Red    = extra / bloated / wrong local geometry near objects", flush=True)

    if args.visualize:
        visualize_two(
            "GT -> Pred (missing view)",
            gt_missing_out,
            "Pred ROI -> GT (extra view)",
            pred_roi_out
        )


if __name__ == "__main__":
    main()