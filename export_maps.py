import os
import torch
import argparse
import re

from scene import Scene
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams
from render import render
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh


# ---------------------------------------------------------------------------
# CONFIG — CHANGE THESE PATHS ONLY
# ---------------------------------------------------------------------------

# 1) Your undistorted COLMAP scene (same for every run)
#COLMAP_UNDISTORTED = r"C:\Users\cwart\Downloads\BlenderDataset\out"#C:\Users\cwart\OneDrive\Desktop\2DGS\COLMAPS\VideoSource\Wide_Light\V2" #C:\Users\cwart\OneDrive\Desktop\2DGS\Evaluation\colmap\undistorted

# 2) Where all configuration folders live
# Example:
#   ...\ShowImages\Output\Normal2DGS\3aae075a-7\point_cloud\iteration_8000
OUTPUT_ROOT = r"C:\Users\cwart\Projects\2d-gaussian-splatting\output" #C:\Users\cwart\OneDrive\Desktop\2DGS\Evaluation\ShowImages\Output

# ---------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"


def find_runs_auto(root: str):
    """
    Supports both:
      root/<run-id>/point_cloud/iteration_XXXX
      root/<config>/<run-id>/point_cloud/iteration_XXXX
    """
    runs = []

    if not os.path.isdir(root):
        print(f"[WARN] Output root not found: {root}")
        return runs

    def scan_run(model_path):
        pc_root = os.path.join(model_path, "point_cloud")
        if not os.path.isdir(pc_root):
            return None

        iterations = []
        for sub in os.listdir(pc_root):
            if sub.startswith("iteration_"):
                try:
                    it = int(sub.split("_")[1])
                    ply = os.path.join(pc_root, sub, "point_cloud.ply")
                    if os.path.isfile(ply):
                        iterations.append(it)
                except ValueError:
                    pass

        if iterations:
            return (model_path, sorted(iterations))
        return None

    for entry in os.listdir(root):
        p = os.path.join(root, entry)
        if not os.path.isdir(p):
            continue

        # Case 1: direct run
        r = scan_run(p)
        if r:
            runs.append(r)
            continue

        # Case 2: config → run
        for sub in os.listdir(p):
            sp = os.path.join(p, sub)
            if not os.path.isdir(sp):
                continue
            r = scan_run(sp)
            if r:
                runs.append(r)

    return runs

def get_source_path_for_run(model_path: str):
    cfg = os.path.join(model_path, "cfg_args")
    if not os.path.isfile(cfg):
        return None, "missing cfg_args"

    txt = open(cfg, "r", encoding="utf-8", errors="ignore").read()

    m = re.search(r"source_path\s*=\s*['\"]([^'\"]+)['\"]", txt)
    if m:
        return m.group(1), "cfg_args: source_path="
    m = re.search(r"--source_path\s+([^\s]+)", txt)
    if m:
        return m.group(1), "cfg_args: --source_path"

    return None, "cfg_args present but source_path not found"


def validate_colmap_scene(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    # Common 2DGS/3DGS expectation: images/ + sparse/ (sometimes sparse/0/)
    images = os.path.join(path, "images")
    sparse = os.path.join(path, "sparse")
    if not os.path.isdir(images):
        return False
    if not os.path.isdir(sparse):
        return False
    return True


def make_param_groups(model_path: str):
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args([])

    sp, sp_src = get_source_path_for_run(model_path)
    if sp is None:
        # fallback (optional)
        sp = COLMAP_UNDISTORTED
        sp_src = "FALLBACK COLMAP_UNDISTORTED"

    if not validate_colmap_scene(sp):
        print(f"[WARN] source_path doesn't look like an undistorted COLMAP scene: {sp}")

    args.source_path = sp
    args.model_path = model_path
    args.images = None
    args.eval = True

    print(f"[RUN] source_path: {args.source_path}")
    print(f"[RUN] source_path_from: {sp_src}")

    mparams = lp.extract(args)
    pipe = pp.extract(args)
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
    runs = find_runs_auto(OUTPUT_ROOT)

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
