import argparse
import json
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
    if method == "uniform":
        return mesh.sample_points_uniformly(number_of_points=n_points)
    if method == "poisson":
        return mesh.sample_points_poisson_disk(number_of_points=n_points)
    raise ValueError(f"Unknown sampling method: {method}")


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


def icp_align(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    threshold: float,
    init_transform: np.ndarray,
    with_scaling: bool
):
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(
        with_scaling=with_scaling
    )

    result = o3d.pipelines.registration.registration_icp(
        source=source_pcd,
        target=target_pcd,
        max_correspondence_distance=threshold,
        init=init_transform,
        estimation_method=estimation,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=200
        ),
    )
    return result


def save_transform_json(path: str, transform: np.ndarray, fitness: float, rmse: float) -> None:
    data = {
        "transformation": transform.tolist(),
        "fitness": float(fitness),
        "inlier_rmse": float(rmse),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align a predicted mesh to a GT mesh using ICP and save the transformed mesh."
    )
    parser.add_argument("--pred_mesh", required=True, type=str, help="Path to predicted .ply mesh")
    parser.add_argument("--gt_mesh", required=True, type=str, help="Path to GT .ply mesh")
    parser.add_argument(
        "--out_mesh",
        required=True,
        type=str,
        help="Path to write the aligned predicted .ply mesh"
    )
    parser.add_argument(
        "--out_transform",
        default=None,
        type=str,
        help="Optional JSON file path for the resulting transform"
    )
    parser.add_argument(
        "--sample_points",
        default=100000,
        type=int,
        help="Number of sampled points per mesh for ICP"
    )
    parser.add_argument(
        "--sampling",
        default="uniform",
        choices=["uniform", "poisson"],
        help="Point sampling method"
    )
    parser.add_argument(
        "--threshold",
        default=0.05,
        type=float,
        help="ICP max correspondence distance"
    )
    parser.add_argument(
        "--with_scaling",
        action="store_true",
        help="Allow ICP to estimate a global scale as well"
    )
    parser.add_argument(
        "--no_center_init",
        action="store_true",
        help="Disable initial center-to-center translation before ICP"
    )
    args = parser.parse_args()

    print(f"[Align] Loading predicted mesh: {args.pred_mesh}", flush=True)
    pred_mesh = load_mesh(args.pred_mesh)

    print(f"[Align] Loading GT mesh: {args.gt_mesh}", flush=True)
    gt_mesh = load_mesh(args.gt_mesh)

    print(f"[Align] Sampling {args.sample_points} points from predicted mesh", flush=True)
    pred_pcd = sample_points(pred_mesh, args.sample_points, method=args.sampling)

    print(f"[Align] Sampling {args.sample_points} points from GT mesh", flush=True)
    gt_pcd = sample_points(gt_mesh, args.sample_points, method=args.sampling)

    init_transform = np.eye(4, dtype=np.float64)

    if not args.no_center_init:
        center_T = compute_center_alignment_transform(pred_pcd, gt_pcd)
        pred_pcd.transform(center_T)
        pred_mesh.transform(center_T)
        init_transform = center_T
        print("[Align] Applied initial center alignment", flush=True)

    print("[Align] Running ICP...", flush=True)
    result = icp_align(
        source_pcd=pred_pcd,
        target_pcd=gt_pcd,
        threshold=args.threshold,
        init_transform=np.eye(4, dtype=np.float64),
        with_scaling=args.with_scaling
    )

    print(f"[Align] ICP fitness     : {result.fitness:.8f}", flush=True)
    print(f"[Align] ICP inlier RMSE : {result.inlier_rmse:.8f}", flush=True)
    print("[Align] ICP transformation:", flush=True)
    print(result.transformation, flush=True)

    pred_mesh.transform(result.transformation)
    total_transform = result.transformation @ init_transform

    out_mesh_path = Path(args.out_mesh)
    out_mesh_path.parent.mkdir(parents=True, exist_ok=True)

    ok = o3d.io.write_triangle_mesh(str(out_mesh_path), pred_mesh)
    if not ok:
        raise IOError(f"Failed to write aligned mesh: {args.out_mesh}")

    print(f"[Align] Wrote aligned mesh to: {args.out_mesh}", flush=True)

    out_transform = args.out_transform
    if out_transform is None:
        out_transform = str(out_mesh_path.with_suffix(".transform.json"))

    Path(out_transform).parent.mkdir(parents=True, exist_ok=True)
    save_transform_json(
        path=out_transform,
        transform=total_transform,
        fitness=result.fitness,
        rmse=result.inlier_rmse
    )
    print(f"[Align] Wrote transform JSON to: {out_transform}", flush=True)


if __name__ == "__main__":
    main()