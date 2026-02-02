import os
import re
import json
import argparse
from pathlib import Path
from PIL import Image

import torch
import torchvision.transforms.functional as tf
from tqdm import tqdm

from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips


# ---------------------------------------------------------------------------
OUTPUT_ROOT_DEFAULT = r"C:\Users\cwart\Projects\2d-gaussian-splatting\output"
# ---------------------------------------------------------------------------


def get_source_path_for_run(model_path: str):
    """Reads cfg_args to extract source_path for traceability (optional for metrics)."""
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


def find_runs_auto(root: str):
    """
    Supports both:
      root/<run-id>/
      root/<config>/<run-id>/
    A 'run' is identified by presence of cfg_args OR point_cloud folder.
    """
    runs = []
    if not os.path.isdir(root):
        print(f"[WARN] Output root not found: {root}")
        return runs

    def is_run_dir(p: str) -> bool:
        if os.path.isfile(os.path.join(p, "cfg_args")):
            return True
        if os.path.isdir(os.path.join(p, "point_cloud")):
            return True
        return False

    for entry in os.listdir(root):
        p = os.path.join(root, entry)
        if not os.path.isdir(p):
            continue

        # Case 1: direct run
        if is_run_dir(p):
            runs.append(p)
            continue

        # Case 2: config → run
        for sub in os.listdir(p):
            sp = os.path.join(p, sub)
            if not os.path.isdir(sp):
                continue
            if is_run_dir(sp):
                runs.append(sp)

    # stable order
    runs = sorted(list(set(runs)), key=lambda x: x.lower())
    return runs


def list_export_iters(run_dir: Path):
    """Return list of (it, export_dir) for export_iter_XXXX folders."""
    out = []
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        m = re.match(r"export_iter_(\d+)$", child.name)
        if not m:
            continue
        it = int(m.group(1))
        # must contain gt/ and renders/
        if (child / "gt").is_dir() and (child / "renders").is_dir():
            out.append((it, child))
    out.sort(key=lambda x: x[0])
    return out


def read_images(renders_dir: Path, gt_dir: Path):
    renders, gts, names = [], [], []
    for fname in sorted(os.listdir(renders_dir)):
        rp = renders_dir / fname
        gp = gt_dir / fname
        if not gp.exists():
            continue
        render = Image.open(rp)
        gt = Image.open(gp)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        names.append(fname)
    return renders, gts, names


def evaluate_export(export_dir: Path, meta: dict, overwrite: bool = False):
    results_path = export_dir / "results.json"
    per_view_path = export_dir / "per_view.json"

    if results_path.exists() and not overwrite:
        print(f"[SKIP] {export_dir} (results.json exists)")
        return "skipped"

    gt_dir = export_dir / "gt"
    renders_dir = export_dir / "renders"
    if not gt_dir.is_dir() or not renders_dir.is_dir():
        print(f"[SKIP] {export_dir} (missing gt/ or renders/)")
        return "skipped"

    renders, gts, names = read_images(renders_dir, gt_dir)
    if not renders:
        print(f"[ERROR] {export_dir} (no matching filenames between renders/ and gt/)")
        return "failed"

    ssims, psnrs, lpipss = [], [], []
    for i in tqdm(range(len(renders)), desc=f"Metrics: {export_dir.parent.name}/{export_dir.name}"):
        ssims.append(ssim(renders[i], gts[i]).item())
        psnrs.append(psnr(renders[i], gts[i]).item())
        lpipss.append(lpips(renders[i], gts[i], net_type="vgg").item())

    results = {
        "SSIM": float(torch.tensor(ssims).mean().item()),
        "PSNR": float(torch.tensor(psnrs).mean().item()),
        "LPIPS": float(torch.tensor(lpipss).mean().item()),
        "meta": meta,
    }
    per_view = {
        "SSIM": dict(zip(names, ssims)),
        "PSNR": dict(zip(names, psnrs)),
        "LPIPS": dict(zip(names, lpipss)),
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(per_view_path, "w", encoding="utf-8") as f:
        json.dump(per_view, f, indent=2)

    print(f"\n[OK] {export_dir}")
    print(f"  SSIM : {results['SSIM']:.7f}")
    print(f"  PSNR : {results['PSNR']:.7f}")
    print(f"  LPIPS: {results['LPIPS']:.7f}\n")
    return "done"


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate all export_iter_* folders under OUTPUT_ROOT; skip if results.json already exists."
    )
    ap.add_argument("--root", "-r", default=OUTPUT_ROOT_DEFAULT,
                    help="Output root containing run folders (default set in script).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute even if results.json exists.")
    ap.add_argument("--continue_on_error", action="store_true",
                    help="Continue even if one export fails.")
    ap.add_argument("--only_latest", action="store_true",
                    help="Evaluate only the latest export_iter_* per run.")
    args = ap.parse_args()

    torch.cuda.set_device(torch.device("cuda:0"))

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"[ERROR] Root does not exist: {root}")

    runs = find_runs_auto(str(root))
    if not runs:
        print("[INFO] No runs found.")
        return 0

    print(f"[INFO] Found {len(runs)} runs.")
    done = skipped = failed = 0

    for run_path in runs:
        run_dir = Path(run_path)
        run_id = run_dir.name
        src_path, src_from = get_source_path_for_run(str(run_dir))

        exports = list_export_iters(run_dir)
        if not exports:
            # no export dirs here -> nothing to do
            continue

        if args.only_latest:
            exports = [exports[-1]]

        for it, export_dir in exports:
            meta = {
                "run_id": run_id,
                "iteration": it,
                "run_dir": str(run_dir),
                "source_path": src_path,
                "source_path_from": src_from,
            }
            status = evaluate_export(export_dir, meta=meta, overwrite=args.overwrite)
            if status == "done":
                done += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                if not args.continue_on_error:
                    print("[STOP] Stopping on first failure. Use --continue_on_error to continue.")
                    print(f"[SUMMARY] done={done}, skipped={skipped}, failed={failed}")
                    return 1

    print(f"[DONE] done={done}, skipped={skipped}, failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
