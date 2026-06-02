#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint, random
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import json
from diff_surfel_rasterization import _C
from utils.lighting_config import load_json, pack_lighting_cfg, save_cfg

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def load_lighting_cfg(path: str) -> dict:
    if not path:
        return {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"lighting cfg json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_lighting_cfg(model_path: str, cfg: dict):
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "cfg_lighting.json"), "w", encoding="utf-8") as f:
        json.dump(cfg or {}, f, indent=2, sort_keys=True)

def compute_gt_highlight_mask(gt_luma):
    q_hi = torch.quantile(gt_luma.flatten(), 0.98)
    q_lo = torch.quantile(gt_luma.flatten(), 0.90)

    thresh_hi = torch.maximum(q_hi, torch.tensor(0.78, device=gt_luma.device))
    thresh_lo = torch.maximum(q_lo, torch.tensor(0.55, device=gt_luma.device))

    return torch.clamp(
        (gt_luma - thresh_lo) / (thresh_hi - thresh_lo + 1e-6),
        0.0,
        1.0
    ).detach()

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_highlight_for_log = 0.0

    train_cams = scene.getTrainCameras()

    highlight_cams = []

    for cam in train_cams:
        gt = cam.original_image.cuda()
        luma = gt.mean(dim=0)

        q95 = torch.quantile(luma.flatten(), 0.95)
        q99 = torch.quantile(luma.flatten(), 0.99)

        # highlight score: bright tail compared to general image brightness
        highlight_contrast = q99 - q95

        if q99 > opt.highlight_threshold or highlight_contrast > 0.08:
            highlight_cams.append(cam)

    print(f"[highlight cams] {len(highlight_cams)} / {len(train_cams)}")

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Material brush-up: periodically force residual fitting through material/light,
        # not through color, opacity, or geometry.
        material_brushup = (
            iteration >= opt.material_brushup_start
            and iteration < opt.material_brushup_end
            and ((iteration - opt.material_brushup_start) % opt.material_brushup_period) < opt.material_brushup_length
        )

        # Pick a random Camera
        if iteration > 8000 and len(highlight_cams) > 0 and random() < 0.7:
            viewpoint_cam = highlight_cams[randint(0, len(highlight_cams) - 1)]
        else:
            if not viewpoint_stack:
                viewpoint_stack = train_cams.copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        gt_image = viewpoint_cam.original_image.cuda()

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)

        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg["rend_normal"]
        surf_normal = render_pkg["surf_normal"]

        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * normal_error.mean()
        dist_loss = lambda_dist * rend_dist.mean()

        # material consistency
        material_loss = torch.tensor(0.0, device="cuda")

        if iteration >= opt.material_cluster_start:
            if iteration == opt.material_cluster_start or iteration % opt.material_cluster_interval == 0:
                gaussians.update_material_clusters(
                    num_clusters=opt.material_cluster_count,
                    iters=8,
                    use_position=True,
                    use_color=True,
                    use_material=True,
                )

            material_loss = gaussians.material_consistency_loss(
                lambda_roughness=opt.lambda_material_roughness,
                lambda_metallic=opt.lambda_material_metallic,
                min_cluster_size=8,
            )

        # simple neutral prior, no variance/evidence
        rough = gaussians.get_roughness
        metal = gaussians.get_metallic

        loss_neutral_roughness = (rough - 0.50).pow(2).mean()

        rough_high_prior = torch.relu(rough - 0.75).pow(2).mean()
        rough_low_prior = torch.relu(0.20 - rough).pow(2).mean()

        metal_extreme_prior = (
            torch.relu(0.02 - metal).pow(2).mean()
            + torch.relu(metal - 0.98).pow(2).mean()
        )

        metal_prior = metal.pow(2).mean()

        render_luma = 0.2126 * image[0:1] + 0.7152 * image[1:2] + 0.0722 * image[2:3]
        gt_luma = 0.2126 * gt_image[0:1] + 0.7152 * gt_image[1:2] + 0.0722 * gt_image[2:3]

        highlight_mask = compute_gt_highlight_mask(gt_luma)

        highlight_loss = torch.tensor(0.0, device="cuda")
        highlight_core_loss = torch.tensor(0.0, device="cuda")
        highlight_core_rough_loss = torch.tensor(0.0, device="cuda")

        if iteration > 12000:
            # broad mask, same as visualized
            highlight_loss = (
                highlight_mask * torch.abs(render_luma - gt_luma)
            ).sum() / (highlight_mask.sum() + 1e-6)

            # small sharp highlight cores
            q_core = torch.quantile(gt_luma.flatten(), 0.995)
            core_thresh = torch.maximum(q_core, torch.tensor(0.85, device=gt_luma.device))

            core_mask = (gt_luma > core_thresh).float().detach()

            missing_core = torch.relu(gt_luma - render_luma).detach()

            highlight_core_loss = (
                core_mask * missing_core.pow(2)
            ).sum() / (core_mask.sum() + 1e-6)

            rough_map = render_pkg.get("rend_roughness", None)

            if rough_map is not None:
                rough_core_target = 0.18
                rough_too_high_core = torch.relu(rough_map - rough_core_target)

                highlight_core_rough_loss = (
                    core_mask * missing_core * rough_too_high_core.pow(2)
                ).sum() / (core_mask.sum() + 1e-6)

        if iteration <= 20000:

            highlight_w = 0.0

            if iteration > 12000:
                highlight_w = min(
                    1.0,
                    (iteration - 12000) / 3000.0
                )

            total_loss = (
                loss
                + dist_loss
                + normal_loss
                + highlight_w * opt.lambda_highlight * highlight_loss
                + highlight_w * opt.lambda_highlight_core * highlight_core_loss
                + highlight_w * opt.lambda_highlight_core_roughness * highlight_core_rough_loss
            )

        else:

            total_loss = (
                0.05 * loss
                + opt.lambda_highlight_late * highlight_loss
                + opt.lambda_highlight_core * highlight_core_loss
                + opt.lambda_highlight_core_roughness * highlight_core_rough_loss
            )

        total_loss.backward()

        # ------------------------------------------------------------
        # staged optimization
        # ------------------------------------------------------------

        if iteration < 7000:

            freeze_names = (
                "roughness",
                "metallic",
            )

        elif iteration < 20000:

            freeze_names = (
                "metallic",
            )

        else:

            freeze_names = (
                "xyz",
                "f_dc",
                "f_rest",
                "opacity",
                "scaling",
                "rotation",
                "ambient",
            )

        for group in gaussians.optimizer.param_groups:

            name = group.get("name", "")

            if name in freeze_names:

                for p in group["params"]:

                    if p.grad is not None:
                        p.grad.zero_()

        # ------------------------------------------------------------
        # material warmup
        # ------------------------------------------------------------

        if iteration < opt.material_warmup_iters:

            for group in gaussians.optimizer.param_groups:

                if group.get("name", "") in ("roughness", "metallic"):

                    for p in group["params"]:

                        if p.grad is not None:
                            p.grad.zero_()

        # ------------------------------------------------------------
        # LR decay after densification
        # ------------------------------------------------------------

        if iteration == opt.densify_until_iter:

            for group in gaussians.optimizer.param_groups:

                if group.get("name", "") in (
                    "roughness",
                    "metallic",
                    "intensity",
                    "ambient"
                ):
                    group["lr"] *= 0.1

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_highlight_for_log = 0.4 * highlight_loss.item() + 0.6 * ema_highlight_for_log

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "highlight": f"{ema_highlight_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/material_loss', material_loss.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/highlight_loss', highlight_loss.item(), iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():
            if network_gui.conn is None:   
                network_gui.try_connect(dataset.render_items)

            if network_gui.conn is not None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam is not None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    
                    SHINY_MIN = 2.0
                    SHINY_MAX = 128.0

                    t = torch.sigmoid(gaussians._metallic).view(-1)[0].item()
                    shininess = SHINY_MIN + (SHINY_MAX - SHINY_MIN) * t

                    viewer_metrics = gaussians.get_viewer_metrics()
                    metrics_dict = {
                        "#": int(gaussians.get_xyz.shape[0]),
                        "loss": float(ema_loss_for_log),
                        "A_eff": viewer_metrics["A_eff"],
                        "R_eff": viewer_metrics["R_eff"],
                        "M_eff": viewer_metrics["M_eff"],
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)

                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)

    # Save cli args
    with open(os.path.join(args.model_path, "cfg_args"), "w", encoding="utf-8") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Save build info (debug)
    try:
        from diff_surfel_rasterization import _C
        info_bytes = _C.get_lighting_build_info()
        if torch.is_tensor(info_bytes):
            info_str = bytes(info_bytes.cpu().tolist()).decode("utf-8", errors="replace")
        else:
            info_str = str(info_bytes)

        with open(os.path.join(args.model_path, "build_info_lighting.txt"), "w", encoding="utf-8") as f:
            f.write(info_str)
    except Exception as e:
        with open(os.path.join(args.model_path, "build_info_lighting_ERROR.txt"), "w", encoding="utf-8") as f:
            f.write(repr(e))

    # ---------------- Lighting config ----------------
    lighting_cfg = getattr(args, "lighting_cfg_dict", {}) or {}

    # Save the runtime lighting config next to cfg_args
    with open(os.path.join(args.model_path, "cfg_lighting.json"), "w", encoding="utf-8") as f:
        json.dump(lighting_cfg, f, indent=2, sort_keys=True)

    # Upload once to CUDA constant memory
    try:
        from diff_surfel_rasterization import _C
        from utils.lighting_config import pack_lighting_cfg  # <- your python helper module
        t = pack_lighting_cfg(lighting_cfg, device="cpu").contiguous()
        _C.set_lighting_config(t)
        torch.cuda.synchronize()
        print("[lighting] upload synchronized OK")
    except Exception as e:
        print(f"[train] lighting cfg upload failed: {e}")

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")

    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument(
    "--lighting_cfg", type=str, default="", help="Path to lighting config JSON (cfg_lighting.json)")
    args = parser.parse_args(sys.argv[1:])

    args.lighting_cfg_dict = load_lighting_cfg(args.lighting_cfg) if args.lighting_cfg else {}
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
