import argparse
import itertools
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d

from scripts.eval_myOwn.my_eval import (
    append_rows_unique_csv,
    compute_bidirectional_distances,
    compute_metrics_from_distances,
    default_metric_fieldnames,
    ensure_dir,
    evaluate_meshes,
    load_mesh,
    print_bbox_info,
    sample_points,
    save_colored_distance_pcd,
    write_json,
)


OURS_RE = re.compile(r"^ours_(\d+)$")


def compute_roi_pred_to_gt_metrics(
    pred_pcd: o3d.geometry.PointCloud,
    gt_pcd: o3d.geometry.PointCloud,
    gt_mesh: o3d.geometry.TriangleMesh,
    roi_padding: float,
    threshold: float,
) -> dict[str, float | str]:
    gt_bbox = gt_mesh.get_axis_aligned_bounding_box()
    min_b = gt_bbox.get_min_bound() - roi_padding
    max_b = gt_bbox.get_max_bound() + roi_padding
    roi_bbox = o3d.geometry.AxisAlignedBoundingBox(min_b, max_b)

    pred_pts = np.asarray(pred_pcd.points, dtype=np.float64)
    if pred_pts.size == 0:
        return {
            "roi_pred_to_gt_mean": "",
            "roi_pred_to_gt_median": "",
            "roi_pred_to_gt_max": "",
            "roi_precision": "",
            "roi_num_pred_points": 0,
        }

    mask = np.all((pred_pts >= min_b) & (pred_pts <= max_b), axis=1)
    pred_roi_pts = pred_pts[mask]

    if pred_roi_pts.shape[0] == 0:
        return {
            "roi_pred_to_gt_mean": "",
            "roi_pred_to_gt_median": "",
            "roi_pred_to_gt_max": "",
            "roi_precision": "",
            "roi_num_pred_points": 0,
        }

    pred_roi_pcd = o3d.geometry.PointCloud()
    pred_roi_pcd.points = o3d.utility.Vector3dVector(pred_roi_pts)

    dists_pred_roi_to_gt = np.asarray(
        pred_roi_pcd.compute_point_cloud_distance(gt_pcd),
        dtype=np.float64
    )

    return {
        "roi_pred_to_gt_mean": float(np.mean(dists_pred_roi_to_gt)),
        "roi_pred_to_gt_median": float(np.median(dists_pred_roi_to_gt)),
        "roi_pred_to_gt_max": float(np.max(dists_pred_roi_to_gt)),
        "roi_precision": float(np.mean(dists_pred_roi_to_gt < threshold)),
        "roi_num_pred_points": int(pred_roi_pts.shape[0]),
    }


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

        iteration = int(m.group(1))
        mesh_path = child / mesh_name
        if mesh_path.is_file():
            if best_iter is None or iteration > best_iter:
                best_iter = iteration
                best_dir = child

    if best_iter is None or best_dir is None:
        return None

    return {
        "name": output_root.name,
        "iteration": str(best_iter),
        "mesh_path": str(best_dir / mesh_name),
    }


def find_highest_export_mesh_with_preference(
    output_root: Path,
    prefer_post: bool = True
) -> dict[str, str] | None:
    mesh_names = ["fuse_post.ply", "fuse.ply"] if prefer_post else ["fuse.ply", "fuse_post.ply"]

    best_result = None
    best_iter = None

    for mesh_name in mesh_names:
        result = find_highest_export_mesh(output_root, mesh_name)
        if result is None:
            continue
        iteration = int(result["iteration"])
        if best_result is None or iteration > best_iter:
            best_result = result
            best_iter = iteration

    return best_result


def find_meshes_in_outputs(base_dir: str, prefer_post: bool = True) -> list[dict[str, str]]:
    base = Path(base_dir)
    if not base.is_dir():
        raise ValueError(f"Base directory does not exist: {base_dir}")

    found = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        result = find_highest_export_mesh_with_preference(child, prefer_post=prefer_post)
        if result is not None:
            found.append(result)

    return found


def sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def visualize_pointcloud(path: str, window_name: str) -> None:
    pcd = o3d.io.read_point_cloud(path)
    o3d.visualization.draw_geometries([pcd], window_name=window_name)


def visualize_meshes(meshes: list[o3d.geometry.TriangleMesh], window_name: str) -> None:
    o3d.visualization.draw_geometries(meshes, window_name=window_name)


def make_translation(tx: float, ty: float, tz: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = tx
    T[1, 3] = ty
    T[2, 3] = tz
    return T


def compute_center_alignment_transform(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud
) -> np.ndarray:
    src_center = np.asarray(source_pcd.get_center(), dtype=np.float64)
    tgt_center = np.asarray(target_pcd.get_center(), dtype=np.float64)
    delta = tgt_center - src_center
    return make_translation(float(delta[0]), float(delta[1]), float(delta[2]))


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


def voxel_downsample_pointcloud(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float
) -> o3d.geometry.PointCloud:
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be > 0")
    return pcd.voxel_down_sample(voxel_size)


def preprocess_pointcloud_for_global_registration(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    pcd_down = voxel_downsample_pointcloud(pcd, voxel_size)

    normal_radius = voxel_size * 2.0
    feature_radius = voxel_size * 5.0

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=100)
    )

    return pcd_down, fpfh


def execute_global_registration_ransac(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    source_fpfh: o3d.pipelines.registration.Feature,
    target_fpfh: o3d.pipelines.registration.Feature,
    voxel_size: float
):
    distance_threshold = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=source_down,
        target=target_down,
        source_feature=source_fpfh,
        target_feature=target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def save_transform_json(path: str, transform: np.ndarray, fitness: float, rmse: float) -> None:
    data = {
        "transformation": transform.tolist(),
        "fitness": float(fitness),
        "inlier_rmse": float(rmse),
    }
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def rigid_icp_align_prediction_to_gt(
    pred_mesh: o3d.geometry.TriangleMesh,
    gt_mesh: o3d.geometry.TriangleMesh,
    sample_points_n: int,
    sampling_method: str,
    threshold: float,
    use_center_init: bool = True,
    max_iteration: int = 200,
    use_gt_bbox_filter: bool = False,
    bbox_padding: float = 0.2,
    use_global_registration: bool = False,
    global_voxel_size: float = 0.05,
) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """
    GT stays fixed.
    Prediction is moved by:
      1) optional center alignment
      2) optional global RANSAC registration
      3) rigid ICP refinement (rotation + translation only, no scaling)
    """
    pred_aligned = o3d.geometry.TriangleMesh(pred_mesh)

    pred_pcd_full = sample_points(pred_aligned, sample_points_n, method=sampling_method)
    gt_pcd_full = sample_points(gt_mesh, sample_points_n, method=sampling_method)

    total_transform = np.eye(4, dtype=np.float64)

    if use_center_init:
        center_T = compute_center_alignment_transform(pred_pcd_full, gt_pcd_full)
        pred_aligned.transform(center_T)
        pred_pcd_full.transform(center_T)
        total_transform = center_T @ total_transform

    if use_gt_bbox_filter:
        gt_bbox = expanded_bbox_from_mesh(gt_mesh, padding=bbox_padding)
        pred_pcd_for_align = filter_pointcloud_by_bbox(pred_pcd_full, gt_bbox)

        num_filtered = len(pred_pcd_for_align.points)
        num_full = len(pred_pcd_full.points)
        print(
            f"[ALIGN] GT-bbox filter kept {num_filtered}/{num_full} predicted sample points "
            f"(padding={bbox_padding})",
            flush=True
        )

        if num_filtered == 0:
            raise ValueError(
                "No predicted sampled points remain inside padded GT bounding box for alignment. "
                "Increase --icp_bbox_padding."
            )
    else:
        pred_pcd_for_align = pred_pcd_full

    gt_pcd_for_align = gt_pcd_full

    global_info = None

    if use_global_registration:
        print(f"[ALIGN] Running global registration with voxel_size={global_voxel_size}", flush=True)

        pred_down, pred_fpfh = preprocess_pointcloud_for_global_registration(
            pred_pcd_for_align, global_voxel_size
        )
        gt_down, gt_fpfh = preprocess_pointcloud_for_global_registration(
            gt_pcd_for_align, global_voxel_size
        )

        print(
            f"[ALIGN] Global registration point counts: "
            f"pred_down={len(pred_down.points)}, gt_down={len(gt_down.points)}",
            flush=True
        )

        global_result = execute_global_registration_ransac(
            source_down=pred_down,
            target_down=gt_down,
            source_fpfh=pred_fpfh,
            target_fpfh=gt_fpfh,
            voxel_size=global_voxel_size,
        )

        pred_aligned.transform(global_result.transformation)
        pred_pcd_full.transform(global_result.transformation)
        pred_pcd_for_align.transform(global_result.transformation)
        total_transform = global_result.transformation @ total_transform

        global_info = {
            "fitness": float(global_result.fitness),
            "inlier_rmse": float(global_result.inlier_rmse),
            "voxel_size": float(global_voxel_size),
            "ransac_max_correspondence_distance": float(global_voxel_size * 1.5),
        }

        print(f"[ALIGN] Global fitness     : {global_result.fitness}", flush=True)
        print(f"[ALIGN] Global inlier RMSE : {global_result.inlier_rmse}", flush=True)

    reg = o3d.pipelines.registration.registration_icp(
        source=pred_pcd_for_align,
        target=gt_pcd_for_align,
        max_correspondence_distance=threshold,
        init=np.eye(4, dtype=np.float64),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=False
        ),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iteration
        ),
    )

    pred_aligned.transform(reg.transformation)
    total_transform = reg.transformation @ total_transform

    info = {
        "transformation": total_transform,
        "fitness": float(reg.fitness),
        "inlier_rmse": float(reg.inlier_rmse),
        "threshold": float(threshold),
        "sample_points": int(sample_points_n),
        "sampling_method": sampling_method,
        "center_init": bool(use_center_init),
        "max_iteration": int(max_iteration),
        "use_gt_bbox_filter": bool(use_gt_bbox_filter),
        "bbox_padding": float(bbox_padding),
        "used_global_registration": bool(use_global_registration),
        "global_registration_fitness": "" if global_info is None else global_info["fitness"],
        "global_registration_inlier_rmse": "" if global_info is None else global_info["inlier_rmse"],
        "global_registration_voxel_size": "" if global_info is None else global_info["voxel_size"],
    }
    return pred_aligned, info


def evaluate_prediction_against_gt(
    pred_mesh_path: str,
    gt_mesh_path: str,
    sample_points_n: int,
    threshold: float,
    sampling_method: str,
    semantics: str,
    align_icp: bool,
    icp_sample_points: int,
    icp_threshold: float,
    icp_center_init: bool,
    icp_max_iteration: int,
    icp_use_gt_bbox_filter: bool,
    icp_bbox_padding: float,
    use_global_registration: bool,
    global_voxel_size: float,
    roi_padding: float,
    save_aligned_mesh_path: str | None = None,
    save_transform_path: str | None = None,
) -> tuple[
    dict,
    o3d.geometry.PointCloud,
    o3d.geometry.PointCloud,
    np.ndarray,
    np.ndarray,
    dict | None,
]:
    """
    a = prediction
    b = GT
    """
    pred_mesh = load_mesh(pred_mesh_path)
    gt_mesh = load_mesh(gt_mesh_path)

    icp_info = None

    if align_icp:
        pred_mesh, icp_info = rigid_icp_align_prediction_to_gt(
            pred_mesh=pred_mesh,
            gt_mesh=gt_mesh,
            sample_points_n=icp_sample_points,
            sampling_method=sampling_method,
            threshold=icp_threshold,
            use_center_init=icp_center_init,
            max_iteration=icp_max_iteration,
            use_gt_bbox_filter=icp_use_gt_bbox_filter,
            bbox_padding=icp_bbox_padding,
            use_global_registration=use_global_registration,
            global_voxel_size=global_voxel_size,
        )

        if save_aligned_mesh_path is not None:
            ensure_dir(str(Path(save_aligned_mesh_path).parent))
            ok = o3d.io.write_triangle_mesh(save_aligned_mesh_path, pred_mesh)
            if not ok:
                raise IOError(f"Failed to write aligned mesh: {save_aligned_mesh_path}")

        if save_transform_path is not None and icp_info is not None:
            save_transform_json(
                path=save_transform_path,
                transform=icp_info["transformation"],
                fitness=icp_info["fitness"],
                rmse=icp_info["inlier_rmse"],
            )

    # Re-sample after alignment for evaluation
    pred_pcd = sample_points(pred_mesh, sample_points_n, method=sampling_method)
    gt_pcd = sample_points(gt_mesh, sample_points_n, method=sampling_method)

    # ROI
    roi_metrics = compute_roi_pred_to_gt_metrics(
        pred_pcd=pred_pcd,
        gt_pcd=gt_pcd,
        gt_mesh=gt_mesh,
        roi_padding=roi_padding,
        threshold=threshold,
    )

    dists_pred_to_gt, dists_gt_to_pred = compute_bidirectional_distances(pred_pcd, gt_pcd)

    metrics = compute_metrics_from_distances(
        dists_a_to_b=dists_pred_to_gt,
        dists_b_to_a=dists_gt_to_pred,
        threshold=threshold,
        label_a=Path(pred_mesh_path).stem,
        label_b=Path(gt_mesh_path).stem,
        semantics=semantics,
    )

    metrics["mesh_path_a"] = pred_mesh_path
    metrics["mesh_path_b"] = gt_mesh_path
    metrics["sample_points"] = int(sample_points_n)
    metrics["sampling_method"] = sampling_method
    metrics["aligned_with_icp"] = bool(align_icp)

    if icp_info is not None:
        metrics["icp_fitness"] = icp_info["fitness"]
        metrics["icp_inlier_rmse"] = icp_info["inlier_rmse"]
        metrics["icp_threshold"] = icp_info["threshold"]
        metrics["icp_sample_points"] = icp_info["sample_points"]
        metrics["used_global_registration"] = icp_info.get("used_global_registration", False)
        metrics["global_registration_fitness"] = icp_info.get("global_registration_fitness", "")
        metrics["global_registration_inlier_rmse"] = icp_info.get("global_registration_inlier_rmse", "")
        metrics["global_registration_voxel_size"] = icp_info.get("global_registration_voxel_size", "")

    if icp_info is None:
        metrics["used_global_registration"] = False
        metrics["global_registration_fitness"] = ""
        metrics["global_registration_inlier_rmse"] = ""
        metrics["global_registration_voxel_size"] = ""

    metrics.update(roi_metrics)

    return metrics, pred_pcd, gt_pcd, dists_pred_to_gt, dists_gt_to_pred, icp_info


def run_pairwise_mode(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)

    print(f"Searching for meshes in: {args.base_dir}")
    found_meshes = find_meshes_in_outputs(args.base_dir, prefer_post=args.prefer_post)

    if not found_meshes:
        print("No meshes found.")
        return

    print("\nFound meshes:")
    labels_to_paths: dict[str, str] = {}
    meshes_for_bbox = {}

    for item in found_meshes:
        label = f"{item['name']}_iter{item['iteration']}"
        labels_to_paths[label] = item["mesh_path"]
        print(f"  {label} -> {item['mesh_path']}")
        meshes_for_bbox[label] = load_mesh(item["mesh_path"])

    print_bbox_info(meshes_for_bbox)

    pairs = list(itertools.combinations(labels_to_paths.items(), 2))
    if not pairs:
        print("Need at least two meshes to compare.")
        return

    csv_path = os.path.join(args.out_dir, "pairwise_metrics.csv")
    rows = []

    print(f"\nComparing {len(pairs)} pairs...\n")
    for (label_a, path_a), (label_b, path_b) in pairs:
        print(f"Comparing {label_a} <-> {label_b}")

        metrics, pcd_a, pcd_b, dists_a_to_b, dists_b_to_a = evaluate_meshes(
            mesh_path_a=path_a,
            mesh_path_b=path_b,
            sample_points_n=args.sample_points,
            threshold=args.fscore_threshold,
            sampling_method=args.sampling,
            semantics="generic"
        )

        metrics["scene"] = "pairwise"
        metrics["mesh_a"] = label_a
        metrics["mesh_b"] = label_b

        print(f"  Chamfer   : {metrics['chamfer']}")
        print(f"  Hausdorff : {metrics['hausdorff']}")
        print(f"  F-score   : {metrics['fscore']} (threshold={args.fscore_threshold})")

        rows.append(metrics)

        pair_json_dir = os.path.join(args.out_dir, "pairwise_json")
        ensure_dir(pair_json_dir)
        json_path = os.path.join(
            pair_json_dir,
            f"{sanitize_name(label_a)}__vs__{sanitize_name(label_b)}.json"
        )
        write_json(json_path, metrics)

        if args.save_colored:
            clip_val = max(
                np.percentile(dists_a_to_b, 95.0),
                np.percentile(dists_b_to_a, 95.0)
            )

            out_a = os.path.join(args.out_dir, f"{sanitize_name(label_a)}_to_{sanitize_name(label_b)}_dist.ply")
            out_b = os.path.join(args.out_dir, f"{sanitize_name(label_b)}_to_{sanitize_name(label_a)}_dist.ply")

            save_colored_distance_pcd(pcd_a, dists_a_to_b, out_a, clip_max=clip_val)
            save_colored_distance_pcd(pcd_b, dists_b_to_a, out_b, clip_max=clip_val)

            print(f"  saved: {out_a}")
            print(f"  saved: {out_b}")

            if args.visualize:
                visualize_pointcloud(out_a, f"{label_a} -> {label_b}")
                visualize_pointcloud(out_b, f"{label_b} -> {label_a}")

        print()

    append_rows_unique_csv(
        csv_path=csv_path,
        rows=rows,
        key_fields=["mesh_a", "mesh_b", "threshold", "sample_points", "sampling_method", "semantics"],
        fieldnames=default_metric_fieldnames()
    )

    print(f"Saved metrics CSV: {csv_path}")


def run_reference_mode(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)

    if not args.reference_mesh:
        raise ValueError("--reference_mesh is required for --mode reference")

    reference_mesh_path = args.reference_mesh
    reference_label = args.reference_name or Path(reference_mesh_path).stem

    print(f"Reference mesh: {reference_mesh_path}")
    print(f"Searching for meshes in: {args.base_dir}")

    found_meshes = find_meshes_in_outputs(args.base_dir, prefer_post=args.prefer_post)
    if not found_meshes:
        print("No meshes found.")
        return

    bbox_meshes = {reference_label: load_mesh(reference_mesh_path)}
    compare_items = []

    print("\nFound meshes:")
    for item in found_meshes:
        label = f"{item['name']}_iter{item['iteration']}"
        compare_items.append((label, item["mesh_path"]))
        print(f"  {label} -> {item['mesh_path']}")
        bbox_meshes[label] = load_mesh(item["mesh_path"])

    print_bbox_info(bbox_meshes)

    ref_csv_name = f"reference_metrics_{sanitize_name(reference_label)}.csv"
    csv_path = os.path.join(args.out_dir, ref_csv_name)
    rows = []

    semantics = "gt_only" if args.gt_only else "pred_gt"

    for label, mesh_path in compare_items:
        print(f"\nComparing predicted mesh {label} against reference {reference_label}")

        aligned_mesh_path = None
        transform_json_path = None

        if args.align_icp:
            align_dir = os.path.join(args.out_dir, "aligned_reference")
            ensure_dir(align_dir)
            aligned_mesh_path = os.path.join(
                align_dir,
                f"{sanitize_name(label)}__aligned_to__{sanitize_name(reference_label)}.ply"
            )
            transform_json_path = os.path.join(
                align_dir,
                f"{sanitize_name(label)}__aligned_to__{sanitize_name(reference_label)}.transform.json"
            )

        metrics, pcd_pred, pcd_ref, dists_pred_to_ref, dists_ref_to_pred, icp_info = evaluate_prediction_against_gt(
            pred_mesh_path=mesh_path,
            gt_mesh_path=reference_mesh_path,
            sample_points_n=args.sample_points,
            threshold=args.fscore_threshold,
            sampling_method=args.sampling,
            semantics=semantics,
            align_icp=args.align_icp,
            icp_sample_points=args.icp_sample_points,
            icp_threshold=args.icp_threshold,
            icp_center_init=not args.icp_no_center_init,
            icp_max_iteration=args.icp_max_iteration,
            icp_use_gt_bbox_filter=args.icp_use_gt_bbox_filter,
            icp_bbox_padding=args.icp_bbox_padding,
            use_global_registration=args.align_global_ransac,
            global_voxel_size=args.global_voxel_size,
            roi_padding=args.roi_padding,
            save_aligned_mesh_path=aligned_mesh_path if args.save_aligned_mesh else None,
            save_transform_path=transform_json_path if args.save_aligned_mesh else None,
        )

        metrics["scene"] = label
        metrics["mesh_a"] = label
        metrics["mesh_b"] = reference_label
        metrics["mesh_path_a"] = mesh_path
        metrics["mesh_path_b"] = reference_mesh_path

        if aligned_mesh_path is not None and args.save_aligned_mesh:
            metrics["aligned_mesh_path"] = aligned_mesh_path
        else:
            metrics["aligned_mesh_path"] = ""

        print(f"  gt->pred mean   : {metrics.get('gt_to_pred_mean', '')}")
        print(f"  gt->pred median : {metrics.get('gt_to_pred_median', '')}")
        print(f"  gt->pred max    : {metrics.get('gt_to_pred_max', '')}")
        print(f"  Recall          : {metrics.get('recall', '')}")

        if args.align_icp and icp_info is not None:
            print(f"  ICP fitness     : {icp_info['fitness']}")
            print(f"  ICP inlier RMSE : {icp_info['inlier_rmse']}")

        if not args.gt_only:
            print(f"  pred->gt mean   : {metrics.get('pred_to_gt_mean', '')}")
            print(f"  Chamfer         : {metrics.get('chamfer', '')}")
            print(f"  Hausdorff       : {metrics.get('hausdorff', '')}")
            print(f"  Precision       : {metrics.get('precision', '')}")
            print(f"  F-score         : {metrics.get('fscore', '')}")

        rows.append(metrics)

        ref_json_dir = os.path.join(args.out_dir, "reference_json")
        ensure_dir(ref_json_dir)
        json_path = os.path.join(
            ref_json_dir,
            f"{sanitize_name(label)}__vs__{sanitize_name(reference_label)}.json"
        )
        write_json(json_path, metrics)

        if args.save_colored:
            if args.gt_only:
                clip_val = np.percentile(dists_ref_to_pred, 95.0)
                out_gt = os.path.join(
                    args.out_dir,
                    f"{sanitize_name(reference_label)}_to_{sanitize_name(label)}_dist.ply"
                )
                save_colored_distance_pcd(pcd_ref, dists_ref_to_pred, out_gt, clip_max=clip_val)
                print(f"  saved: {out_gt}")
                if args.visualize:
                    visualize_pointcloud(out_gt, f"{reference_label} -> {label}")
            else:
                clip_val = max(
                    np.percentile(dists_pred_to_ref, 95.0),
                    np.percentile(dists_ref_to_pred, 95.0)
                )

                out_pred = os.path.join(
                    args.out_dir,
                    f"{sanitize_name(label)}_to_{sanitize_name(reference_label)}_dist.ply"
                )
                out_ref = os.path.join(
                    args.out_dir,
                    f"{sanitize_name(reference_label)}_to_{sanitize_name(label)}_dist.ply"
                )

                save_colored_distance_pcd(pcd_pred, dists_pred_to_ref, out_pred, clip_max=clip_val)
                save_colored_distance_pcd(pcd_ref, dists_ref_to_pred, out_ref, clip_max=clip_val)

                print(f"  saved: {out_pred}")
                print(f"  saved: {out_ref}")

                if args.visualize:
                    visualize_pointcloud(out_pred, f"{label} -> {reference_label}")
                    visualize_pointcloud(out_ref, f"{reference_label} -> {label}")

    append_rows_unique_csv(
        csv_path=csv_path,
        rows=rows,
        key_fields=["mesh_a", "mesh_b", "threshold", "sample_points", "sampling_method", "semantics", "aligned_with_icp"],
        fieldnames=default_metric_fieldnames() + [
            "roi_pred_to_gt_mean",
            "roi_pred_to_gt_median",
            "roi_pred_to_gt_max",
            "roi_precision",
            "roi_num_pred_points",
            "aligned_with_icp",
            "icp_fitness",
            "icp_inlier_rmse",
            "icp_threshold",
            "icp_sample_points",
            "aligned_mesh_path",
            "used_global_registration",
            "global_registration_fitness",
            "global_registration_inlier_rmse",
            "global_registration_voxel_size",
        ]
    )

    print(f"\nSaved metrics CSV: {csv_path}")


def find_scene_gt_mesh(scene_dir: Path, gt_rel_path: str) -> str:
    gt_path = scene_dir / gt_rel_path
    if not gt_path.is_file():
        raise FileNotFoundError(f"GT mesh not found: {gt_path}")
    return str(gt_path)


def find_scene_pred_mesh(output_path: str, scene_name: str, iteration: int | None, prefer_post: bool) -> str:
    scene_output_dir = Path(output_path) / scene_name
    train_dir = scene_output_dir / "train"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing train dir: {train_dir}")

    if iteration is not None:
        candidate_names = ["fuse_post.ply", "fuse.ply"] if prefer_post else ["fuse.ply", "fuse_post.ply"]
        ours_dir = train_dir / f"ours_{iteration}"
        for mesh_name in candidate_names:
            mesh_path = ours_dir / mesh_name
            if mesh_path.is_file():
                return str(mesh_path)
        raise FileNotFoundError(f"No predicted mesh found in {ours_dir}")

    result = find_highest_export_mesh_with_preference(scene_output_dir, prefer_post=prefer_post)
    if result is None:
        raise FileNotFoundError(f"No predicted mesh found for scene output: {scene_output_dir}")
    return result["mesh_path"]


def run_blender_eval_mode(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)

    if not args.blender_data:
        raise ValueError("--blender_data is required for --mode blender_eval")
    if not args.output_path:
        raise ValueError("--output_path is required for --mode blender_eval")

    blender_root = Path(args.blender_data)
    if not blender_root.is_dir():
        raise ValueError(f"Blender data root does not exist: {args.blender_data}")

    if args.scenes:
        scene_names = args.scenes
    else:
        scene_names = sorted([
            child.name for child in blender_root.iterdir()
            if child.is_dir()
        ])

    rows = []
    csv_path = os.path.join(args.out_dir, "blender_eval_metrics.csv")

    for scene_name in scene_names:
        print(f"\n=== Scene: {scene_name} ===")
        scene_dir = blender_root / scene_name

        gt_mesh = find_scene_gt_mesh(scene_dir, args.gt_rel_path)
        pred_mesh = find_scene_pred_mesh(
            output_path=args.output_path,
            scene_name=scene_name,
            iteration=args.iteration,
            prefer_post=args.prefer_post
        )

        print(f"GT   : {gt_mesh}")
        print(f"Pred : {pred_mesh}")

        scene_out_dir = os.path.join(args.out_dir, scene_name)
        ensure_dir(scene_out_dir)

        aligned_mesh_path = None
        transform_json_path = None

        if args.align_icp:
            aligned_mesh_path = os.path.join(scene_out_dir, "pred_aligned_to_gt.ply")
            transform_json_path = os.path.join(scene_out_dir, "pred_aligned_to_gt.transform.json")

        metrics, pcd_pred, pcd_gt, dists_pred_to_gt, dists_gt_to_pred, icp_info = evaluate_prediction_against_gt(
            pred_mesh_path=pred_mesh,
            gt_mesh_path=gt_mesh,
            sample_points_n=args.sample_points,
            threshold=args.fscore_threshold,
            sampling_method=args.sampling,
            semantics="gt_only",
            align_icp=args.align_icp,
            icp_sample_points=args.icp_sample_points,
            icp_threshold=args.icp_threshold,
            icp_center_init=not args.icp_no_center_init,
            icp_max_iteration=args.icp_max_iteration,
            icp_use_gt_bbox_filter=args.icp_use_gt_bbox_filter,
            icp_bbox_padding=args.icp_bbox_padding,
            use_global_registration=args.align_global_ransac,
            global_voxel_size=args.global_voxel_size,
            roi_padding=args.roi_padding,
            save_aligned_mesh_path=aligned_mesh_path if args.save_aligned_mesh else None,
            save_transform_path=transform_json_path if args.save_aligned_mesh else None,
        )

        metrics["scene"] = scene_name
        metrics["mesh_a"] = Path(pred_mesh).stem
        metrics["mesh_b"] = Path(gt_mesh).stem
        metrics["mesh_path_a"] = pred_mesh
        metrics["mesh_path_b"] = gt_mesh
        metrics["aligned_mesh_path"] = aligned_mesh_path if (aligned_mesh_path and args.save_aligned_mesh) else ""

        print(f"  gt->pred mean   : {metrics.get('gt_to_pred_mean', '')}")
        print(f"  gt->pred median : {metrics.get('gt_to_pred_median', '')}")
        print(f"  gt->pred max    : {metrics.get('gt_to_pred_max', '')}")
        print(f"  Recall          : {metrics.get('recall', '')}")

        if args.align_icp and icp_info is not None:
            print(f"  ICP fitness     : {icp_info['fitness']}")
            print(f"  ICP inlier RMSE : {icp_info['inlier_rmse']}")

        print("  pred->gt / chamfer / hausdorff / precision / fscore are disabled in gt_only mode")

        write_json(os.path.join(scene_out_dir, "metrics.json"), metrics)

        if args.save_colored:
            clip_val = np.percentile(dists_gt_to_pred, 95.0)
            out_gt = os.path.join(scene_out_dir, "gt_to_pred_dist.ply")
            save_colored_distance_pcd(pcd_gt, dists_gt_to_pred, out_gt, clip_max=clip_val)
            print(f"  saved: {out_gt}")

            if args.visualize:
                visualize_pointcloud(out_gt, f"{scene_name}: gt -> pred")

        rows.append(metrics)

    append_rows_unique_csv(
        csv_path=csv_path,
        rows=rows,
        key_fields=["scene", "mesh_path_a", "mesh_path_b", "threshold", "sample_points", "sampling_method", "semantics", "aligned_with_icp"],
        fieldnames=default_metric_fieldnames() + [
            "aligned_with_icp",
            "icp_fitness",
            "icp_inlier_rmse",
            "icp_threshold",
            "icp_sample_points",
            "aligned_mesh_path",
            "used_global_registration",
            "global_registration_fitness",
            "global_registration_inlier_rmse",
            "global_registration_voxel_size",
        ]
    )

    print(f"\nSaved metrics CSV: {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="General mesh comparison and Blender GT evaluation tool with optional rigid ICP alignment."
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["pairwise", "reference", "blender_eval"],
        help="Evaluation mode"
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        default="./output",
        help="Base output directory containing run folders"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output",
        help="Output root for blender_eval mode"
    )
    parser.add_argument(
        "--blender_data",
        type=str,
        default=None,
        help="Blender dataset root for blender_eval mode"
    )
    parser.add_argument(
        "--gt_rel_path",
        type=str,
        default="gt/gt_eval_mesh.ply",
        help="Relative GT mesh path inside each scene folder"
    )
    parser.add_argument(
        "--reference_mesh",
        type=str,
        default=None,
        help="Reference mesh path for reference mode"
    )
    parser.add_argument(
        "--reference_name",
        type=str,
        default=None,
        help="Optional label for reference mesh"
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Specific iteration for blender_eval mode; default uses highest found"
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Optional scene names for blender_eval mode"
    )
    parser.add_argument(
        "--sample_points",
        type=int,
        default=100000,
        help="Number of sampled points per mesh for evaluation"
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default="uniform",
        choices=["uniform", "poisson"],
        help="Mesh point sampling method"
    )
    parser.add_argument(
        "--fscore_threshold",
        type=float,
        default=0.01,
        help="Distance threshold for recall / F-score style thresholds"
    )
    parser.add_argument(
        "--prefer_post",
        action="store_true",
        help="Prefer fuse_post.ply over fuse.ply"
    )
    parser.add_argument(
        "--gt_only",
        action="store_true",
        help="For reference mode: ignore pred-only extra geometry and evaluate GT coverage only"
    )

    # ICP
    parser.add_argument(
        "--align_icp",
        action="store_true",
        help="Align prediction to GT with rigid ICP before evaluation"
    )
    parser.add_argument(
        "--icp_sample_points",
        type=int,
        default=100000,
        help="Number of sampled points per mesh used for ICP"
    )
    parser.add_argument(
        "--icp_threshold",
        type=float,
        default=0.05,
        help="Rigid ICP max correspondence distance"
    )
    parser.add_argument(
        "--icp_no_center_init",
        action="store_true",
        help="Disable initial center-to-center translation before ICP"
    )
    parser.add_argument(
        "--icp_max_iteration",
        type=int,
        default=200,
        help="Maximum ICP iterations"
    )
    parser.add_argument(
        "--save_aligned_mesh",
        action="store_true",
        help="Save the ICP-aligned prediction mesh and transform JSON"
    )

    parser.add_argument(
        "--save_colored",
        action="store_true",
        help="Save distance-colored point clouds"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize saved colored point clouds in Open3D"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="mesh_compare_out",
        help="Output directory"
    )
    parser.add_argument(
        "--icp_use_gt_bbox_filter",
        action="store_true",
        help="Restrict prediction points used for ICP to a padded GT bounding box"
    )
    parser.add_argument(
        "--icp_bbox_padding",
        type=float,
        default=0.2,
        help="Padding added to GT bounding box for ICP point filtering"
    )
    parser.add_argument(
        "--align_global_ransac",
        action="store_true",
        help="Run FPFH + RANSAC global registration before rigid ICP"
    )
    parser.add_argument(
        "--global_voxel_size",
        type=float,
        default=0.05,
        help="Voxel size used for global registration downsampling and FPFH features"
    )
    parser.add_argument(
        "--roi_padding",
        type=float,
        default=0.2,
        help="Padding added to GT object bounding box for ROI-restricted pred->gt metrics"
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "pairwise":
        run_pairwise_mode(args)
    elif args.mode == "reference":
        run_reference_mode(args)
    elif args.mode == "blender_eval":
        run_blender_eval_mode(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()