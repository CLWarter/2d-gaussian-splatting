import csv
import json
import os
from pathlib import Path
from typing import Any

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
    if n_points <= 0:
        raise ValueError("n_points must be > 0")

    method = method.lower()
    if method == "uniform":
        return mesh.sample_points_uniformly(number_of_points=n_points)
    if method == "poisson":
        return mesh.sample_points_poisson_disk(number_of_points=n_points)

    raise ValueError(f"Unknown sampling method: {method}")


def compute_bidirectional_distances(
    pcd_a: o3d.geometry.PointCloud,
    pcd_b: o3d.geometry.PointCloud
) -> tuple[np.ndarray, np.ndarray]:
    dists_a_to_b = np.asarray(
        pcd_a.compute_point_cloud_distance(pcd_b),
        dtype=np.float64
    )
    dists_b_to_a = np.asarray(
        pcd_b.compute_point_cloud_distance(pcd_a),
        dtype=np.float64
    )
    return dists_a_to_b, dists_b_to_a


def chamfer_distance(dists_a_to_b: np.ndarray, dists_b_to_a: np.ndarray) -> float:
    return float(np.mean(dists_a_to_b) + np.mean(dists_b_to_a))


def hausdorff_distance(dists_a_to_b: np.ndarray, dists_b_to_a: np.ndarray) -> float:
    return float(max(np.max(dists_a_to_b), np.max(dists_b_to_a)))


def precision_recall_fscore(
    dists_pred_to_gt: np.ndarray,
    dists_gt_to_pred: np.ndarray,
    threshold: float
) -> tuple[float, float, float]:
    precision = float(np.mean(dists_pred_to_gt < threshold))
    recall = float(np.mean(dists_gt_to_pred < threshold))
    if precision + recall == 0.0:
        return precision, recall, 0.0
    fscore = 2.0 * precision * recall / (precision + recall)
    return precision, recall, fscore


def distances_to_colors(dists: np.ndarray, clip_max: float | None = None) -> np.ndarray:
    vals = dists.copy()
    if len(vals) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    if clip_max is None:
        clip_max = np.percentile(vals, 95.0)

    clip_max = max(float(clip_max), 1e-12)
    vals = np.clip(vals / clip_max, 0.0, 1.0)

    colors = np.zeros((len(vals), 3), dtype=np.float64)

    for i, v in enumerate(vals):
        if v < 0.25:
            t = v / 0.25
            colors[i] = [0.0, t, 1.0]
        elif v < 0.5:
            t = (v - 0.25) / 0.25
            colors[i] = [0.0, 1.0, 1.0 - t]
        elif v < 0.75:
            t = (v - 0.5) / 0.25
            colors[i] = [t, 1.0, 0.0]
        else:
            t = (v - 0.75) / 0.25
            colors[i] = [1.0, 1.0 - t, 0.0]

    return colors


def save_colored_distance_pcd(
    source_pcd: o3d.geometry.PointCloud,
    dists: np.ndarray,
    out_path: str,
    clip_max: float | None = None
) -> None:
    pcd = o3d.geometry.PointCloud(source_pcd)
    pcd.colors = o3d.utility.Vector3dVector(
        distances_to_colors(dists, clip_max=clip_max)
    )
    ok = o3d.io.write_point_cloud(out_path, pcd)
    if not ok:
        raise IOError(f"Failed to write point cloud: {out_path}")


def print_bbox_info(meshes: dict[str, o3d.geometry.TriangleMesh]) -> None:
    print("\nBounding box extents:")
    for name, mesh in meshes.items():
        bbox = mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        print(f"  {name}: extent = {extent}")
    print("\nMake sure all meshes are aligned and in the same coordinate system.\n")


def compute_metrics_from_distances(
    dists_a_to_b: np.ndarray,
    dists_b_to_a: np.ndarray,
    threshold: float,
    label_a: str,
    label_b: str,
    semantics: str = "generic"
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "mesh_a": label_a,
        "mesh_b": label_b,
        "threshold": float(threshold),
        "semantics": semantics,
    }

    # a = pred, b = gt
    if semantics == "pred_gt":
        metrics["mean_a_to_b"] = float(np.mean(dists_a_to_b))
        metrics["mean_b_to_a"] = float(np.mean(dists_b_to_a))
        metrics["median_a_to_b"] = float(np.median(dists_a_to_b))
        metrics["median_b_to_a"] = float(np.median(dists_b_to_a))
        metrics["max_a_to_b"] = float(np.max(dists_a_to_b))
        metrics["max_b_to_a"] = float(np.max(dists_b_to_a))
        metrics["chamfer"] = chamfer_distance(dists_a_to_b, dists_b_to_a)
        metrics["hausdorff"] = hausdorff_distance(dists_a_to_b, dists_b_to_a)

        precision, recall, fscore = precision_recall_fscore(
            dists_pred_to_gt=dists_a_to_b,
            dists_gt_to_pred=dists_b_to_a,
            threshold=threshold
        )
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["fscore"] = fscore
        metrics["pred_to_gt_mean"] = metrics["mean_a_to_b"]
        metrics["gt_to_pred_mean"] = metrics["mean_b_to_a"]
        metrics["gt_to_pred_median"] = metrics["median_b_to_a"]
        metrics["gt_to_pred_max"] = metrics["max_b_to_a"]
        return metrics

    # a = pred, b = gt, but we IGNORE pred-only extra geometry like floor
    if semantics == "gt_only":
        metrics["mean_a_to_b"] = ""
        metrics["mean_b_to_a"] = float(np.mean(dists_b_to_a))
        metrics["median_a_to_b"] = ""
        metrics["median_b_to_a"] = float(np.median(dists_b_to_a))
        metrics["max_a_to_b"] = ""
        metrics["max_b_to_a"] = float(np.max(dists_b_to_a))

        metrics["pred_to_gt_mean"] = ""
        metrics["gt_to_pred_mean"] = metrics["mean_b_to_a"]
        metrics["gt_to_pred_median"] = metrics["median_b_to_a"]
        metrics["gt_to_pred_max"] = metrics["max_b_to_a"]

        # These symmetric metrics are intentionally disabled because they would
        # penalize extra output geometry like floor that is absent in GT.
        metrics["chamfer"] = ""
        metrics["hausdorff"] = ""
        metrics["precision"] = ""
        metrics["fscore"] = ""

        # recall stays meaningful: fraction of GT covered by prediction
        metrics["recall"] = float(np.mean(dists_b_to_a < threshold))
        return metrics

    # generic symmetric case
    metrics["mean_a_to_b"] = float(np.mean(dists_a_to_b))
    metrics["mean_b_to_a"] = float(np.mean(dists_b_to_a))
    metrics["median_a_to_b"] = float(np.median(dists_a_to_b))
    metrics["median_b_to_a"] = float(np.median(dists_b_to_a))
    metrics["max_a_to_b"] = float(np.max(dists_a_to_b))
    metrics["max_b_to_a"] = float(np.max(dists_b_to_a))
    metrics["chamfer"] = chamfer_distance(dists_a_to_b, dists_b_to_a)
    metrics["hausdorff"] = hausdorff_distance(dists_a_to_b, dists_b_to_a)

    precision, recall, fscore = precision_recall_fscore(
        dists_pred_to_gt=dists_a_to_b,
        dists_gt_to_pred=dists_b_to_a,
        threshold=threshold
    )
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["fscore"] = fscore
    metrics["pred_to_gt_mean"] = ""
    metrics["gt_to_pred_mean"] = ""
    metrics["gt_to_pred_median"] = ""
    metrics["gt_to_pred_max"] = ""
    return metrics


def evaluate_pointclouds(
    label_a: str,
    pcd_a: o3d.geometry.PointCloud,
    label_b: str,
    pcd_b: o3d.geometry.PointCloud,
    threshold: float,
    semantics: str = "generic"
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    dists_a_to_b, dists_b_to_a = compute_bidirectional_distances(pcd_a, pcd_b)
    metrics = compute_metrics_from_distances(
        dists_a_to_b=dists_a_to_b,
        dists_b_to_a=dists_b_to_a,
        threshold=threshold,
        label_a=label_a,
        label_b=label_b,
        semantics=semantics
    )
    return metrics, dists_a_to_b, dists_b_to_a


def evaluate_meshes(
    mesh_path_a: str,
    mesh_path_b: str,
    sample_points_n: int,
    threshold: float,
    sampling_method: str = "uniform",
    semantics: str = "generic"
) -> tuple[
    dict[str, Any],
    o3d.geometry.PointCloud,
    o3d.geometry.PointCloud,
    np.ndarray,
    np.ndarray
]:
    mesh_a = load_mesh(mesh_path_a)
    mesh_b = load_mesh(mesh_path_b)

    pcd_a = sample_points(mesh_a, sample_points_n, method=sampling_method)
    pcd_b = sample_points(mesh_b, sample_points_n, method=sampling_method)

    metrics, dists_a_to_b, dists_b_to_a = evaluate_pointclouds(
        label_a=Path(mesh_path_a).stem,
        pcd_a=pcd_a,
        label_b=Path(mesh_path_b).stem,
        pcd_b=pcd_b,
        threshold=threshold,
        semantics=semantics
    )

    metrics["mesh_path_a"] = mesh_path_a
    metrics["mesh_path_b"] = mesh_path_b
    metrics["sample_points"] = int(sample_points_n)
    metrics["sampling_method"] = sampling_method

    return metrics, pcd_a, pcd_b, dists_a_to_b, dists_b_to_a


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, data: dict[str, Any]) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def append_rows_unique_csv(
    csv_path: str,
    rows: list[dict[str, Any]],
    key_fields: list[str],
    fieldnames: list[str]
) -> None:
    ensure_dir(str(Path(csv_path).parent))

    existing_keys: set[tuple[Any, ...]] = set()
    file_exists = os.path.isfile(csv_path)

    if file_exists:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_keys.add(tuple(row.get(k, "") for k in key_fields))

    new_rows = []
    for row in rows:
        normalized = {}
        for field in fieldnames:
            normalized[field] = row.get(field, "")

        key = tuple(str(normalized.get(k, "")) for k in key_fields)
        if key not in existing_keys:
            new_rows.append(normalized)
            existing_keys.add(key)

    mode = "a" if file_exists else "w"
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in new_rows:
            writer.writerow(row)


def default_metric_fieldnames() -> list[str]:
    return [
        "scene",
        "mesh_a",
        "mesh_b",
        "mesh_path_a",
        "mesh_path_b",
        "mean_a_to_b",
        "mean_b_to_a",
        "median_a_to_b",
        "median_b_to_a",
        "max_a_to_b",
        "max_b_to_a",
        "pred_to_gt_mean",
        "gt_to_pred_mean",
        "gt_to_pred_median",
        "gt_to_pred_max",
        "chamfer",
        "hausdorff",
        "precision",
        "recall",
        "fscore",
        "threshold",
        "sample_points",
        "sampling_method",
        "semantics",
    ]