import os
import json
import argparse
import numpy as np
import open3d as o3d


def sample_mesh_points(mesh, n_points):
    if not mesh.has_triangles():
        raise ValueError("Mesh has no triangles.")
    pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    return np.asarray(pcd.points, dtype=np.float64)


def nearest_distances(src_pts, dst_pts):
    dst_pcd = o3d.geometry.PointCloud()
    dst_pcd.points = o3d.utility.Vector3dVector(dst_pts)
    kdtree = o3d.geometry.KDTreeFlann(dst_pcd)

    dists = np.empty(len(src_pts), dtype=np.float64)
    for i, p in enumerate(src_pts):
        _, idx, dist2 = kdtree.search_knn_vector_3d(p, 1)
        dists[i] = np.sqrt(dist2[0]) if len(dist2) > 0 else np.inf
    return dists


def compute_metrics(gt_pts, pred_pts, threshold):
    gt_to_pred = nearest_distances(gt_pts, pred_pts)
    pred_to_gt = nearest_distances(pred_pts, gt_pts)

    chamfer = gt_to_pred.mean() + pred_to_gt.mean()
    hausdorff = max(gt_to_pred.max(), pred_to_gt.max())

    recall = float((gt_to_pred < threshold).mean())
    precision = float((pred_to_gt < threshold).mean())

    if precision + recall > 0.0:
        fscore = 2.0 * precision * recall / (precision + recall)
    else:
        fscore = 0.0

    return {
        "gt_to_pred_mean": float(gt_to_pred.mean()),
        "pred_to_gt_mean": float(pred_to_gt.mean()),
        "chamfer": float(chamfer),
        "hausdorff": float(hausdorff),
        "precision": precision,
        "recall": recall,
        "fscore": float(fscore),
        "threshold": float(threshold),
        "num_gt_points": int(len(gt_pts)),
        "num_pred_points": int(len(pred_pts)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_mesh", required=True, type=str)
    parser.add_argument("--pred_mesh", required=True, type=str)
    parser.add_argument("--scene_name", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--sample_points", default=200000, type=int)
    parser.add_argument("--threshold", default=0.01, type=float)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Eval] Loading GT mesh: {args.gt_mesh}", flush=True)
    gt_mesh = o3d.io.read_triangle_mesh(args.gt_mesh)

    print(f"[Eval] Loading predicted mesh: {args.pred_mesh}", flush=True)
    pred_mesh = o3d.io.read_triangle_mesh(args.pred_mesh)

    if gt_mesh.is_empty():
        raise ValueError("GT mesh is empty.")
    if pred_mesh.is_empty():
        raise ValueError("Predicted mesh is empty.")

    gt_mesh.compute_vertex_normals()
    pred_mesh.compute_vertex_normals()

    print(f"[Eval] Sampling {args.sample_points} points from GT mesh", flush=True)
    gt_pts = sample_mesh_points(gt_mesh, args.sample_points)

    print(f"[Eval] Sampling {args.sample_points} points from predicted mesh", flush=True)
    pred_pts = sample_mesh_points(pred_mesh, args.sample_points)

    print("[Eval] Computing nearest-neighbor distances", flush=True)
    metrics = compute_metrics(gt_pts, pred_pts, args.threshold)
    metrics["scene_name"] = args.scene_name
    metrics["gt_mesh"] = args.gt_mesh
    metrics["pred_mesh"] = args.pred_mesh

    out_json = os.path.join(args.output_dir, "metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("[Eval] Results:", flush=True)
    for k, v in metrics.items():
        print(f"  {k}: {v}", flush=True)


if __name__ == "__main__":
    main()