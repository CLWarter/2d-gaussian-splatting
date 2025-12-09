import os
import torch
import argparse

from scene import Scene
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams
from render import render
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh


# ---------------------------------------------------------------------------
# CONFIG — CHANGE THESE PATHS ONLY
# ---------------------------------------------------------------------------

# 1) Your undistorted COLMAP scene (same for every run)
COLMAP_UNDISTORTED = r"C:\Users\cwart\OneDrive\Desktop\2DGS\Evaluation\colmap\undistorted"

# 2) Where all configuration folders live
# Example:
#   ...\ShowImages\Output\Normal2DGS\3aae075a-7\point_cloud\iteration_8000
OUTPUT_ROOT = r"C:\Users\cwart\OneDrive\Desktop\2DGS\Evaluation\ShowImages\Output"

# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"


def find_runs_two_levels(root: str):
    """
    Find runs under layout:
        root/<config>/<run-id>/point_cloud/iteration_XXXX
    Returns list of (model_path, [iteration_numbers])
    """
    runs = []

    if not os.path.isdir(root):
        print(f"[WARN] Output root not found: {root}")
        return runs

    for config_name in os.listdir(root):
        config_dir = os.path.join(root, config_name)
        if not os.path.isdir(config_dir):
            continue

        for run_id in os.listdir(config_dir):
            model_path = os.path.join(config_dir, run_id)
            pc_root = os.path.join(model_path, "point_cloud")
            if not os.path.isdir(pc_root):
                continue

            iterations = []
            for sub in os.listdir(pc_root):
                if not sub.startswith("iteration_"):
                    continue
                try:
                    it = int(sub.split("_")[1])
                    ply = os.path.join(pc_root, sub, "point_cloud.ply")
                    if os.path.isfile(ply):
                        iterations.append(it)
                except Exception:
                    continue

            if iterations:
                runs.append((model_path, sorted(iterations)))

    return runs


def make_param_groups(model_path: str):
    """
    Build lp/pp + extracted args in the same way train.py does,
    but with our own source_path/model_path/eval settings.
    """
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)

    # parse empty CLI → use defaults
    args = parser.parse_args([])

    # override the fields we care about
    args.source_path = COLMAP_UNDISTORTED
    args.model_path = model_path
    args.images = None
    args.eval = True

    mparams = lp.extract(args)   # has .source_path, .model_path, .sh_degree, ...
    pipe    = pp.extract(args)

    return mparams, pipe


def export_for_run(model_path: str, it: int):
    """
    Load Gaussian scene at given iteration and export maps.
    """
    print(f"\n[RUN] Model path: {model_path}")
    print(f"[RUN] Loading iteration {it}")

    mparams, pipe = make_param_groups(model_path)

    # Load scene
    gaussians = GaussianModel(sh_degree=mparams.sh_degree)
    scene = Scene(mparams, gaussians, load_iteration=it, shuffle=False)

    # Choose viewpoints
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    if test_cams:
        viewpoints = test_cams
        print(f"[RUN] Using {len(viewpoints)} TEST cameras.")
    else:
        viewpoints = train_cams
        print(f"[RUN] No test cameras found → using {len(viewpoints)} TRAIN cameras.")

    # Export directory next to run folder:
    export_root = os.path.join(model_path, f"export_iter_{it}")
    print(f"[RUN] Exporting maps to: {export_root}")

    extractor = GaussianExtractor(scene.gaussians, render, pipe)
    extractor.reconstruction(viewpoints)
    extractor.export_image(export_root)

    print("[RUN] Export complete.")


def main():
    print(f"[INFO] Searching for runs in: {OUTPUT_ROOT}")
    runs = find_runs_two_levels(OUTPUT_ROOT)

    if not runs:
        print("[INFO] No runs found.")
        return

    print(f"[INFO] Found {len(runs)} runs total:")
    for model_path, iters in runs:
        print(f"  - {model_path}, iterations: {iters}")

    for model_path, iters in runs:
        last_it = iters[-1]  # use final iteration for each run
        export_for_run(model_path, last_it)


if __name__ == "__main__":
    main()
