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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.render_utils import generate_path, create_videos

import open3d as o3d

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    parser.add_argument("--lighting_cfg", type=str, default="", help="Path to cfg_lighting.json")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)


    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)    
    
    if not args.skip_train:
        print("export training images ...")
        os.makedirs(train_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTrainCameras())
        gaussExtractor.export_image(train_dir)
        
    
    if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
        print("export rendered testing images ...")
        os.makedirs(test_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTestCameras())
        gaussExtractor.export_image(test_dir)
    
    
    if args.render_path:
        print("render videos ...")
        traj_dir = os.path.join(args.model_path, 'traj', "ours_{}".format(scene.loaded_iter))
        os.makedirs(traj_dir, exist_ok=True)
        n_fames = 240
        cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
        gaussExtractor.reconstruction(cam_traj)
        gaussExtractor.export_image(traj_dir)
        create_videos(base_dir=traj_dir,
                    input_dir=traj_dir, 
                    out_name='render_traj', 
                    num_frames=n_fames)

    import time

    if not args.skip_mesh:
        print("export mesh ...", flush=True)
        os.makedirs(train_dir, exist_ok=True)

        gaussExtractor.gaussians.active_sh_degree = 0
        print("active_sh_degree set to 0", flush=True)

        cams = scene.getTrainCameras()
        print("num train cams:", len(cams), flush=True)

        t0 = time.time()
        gaussExtractor.reconstruction(cams)
        print(f"reconstruction finished in {time.time() - t0:.2f}s", flush=True)

        if args.unbounded:
            name = 'fuse_unbounded.ply'
            print("starting unbounded mesh extraction ...", flush=True)
            t1 = time.time()
            mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
            print(f"unbounded extraction finished in {time.time() - t1:.2f}s", flush=True)
        else:
            name = 'fuse.ply'
            depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0 else args.depth_trunc
            voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
            sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc

            print("depth_trunc =", depth_trunc, flush=True)
            print("voxel_size  =", voxel_size, flush=True)
            print("sdf_trunc   =", sdf_trunc, flush=True)

            print("starting bounded mesh extraction ...", flush=True)
            t1 = time.time()
            mesh = gaussExtractor.extract_mesh_bounded(
                voxel_size=voxel_size,
                sdf_trunc=sdf_trunc,
                depth_trunc=depth_trunc
            )
            print(f"bounded extraction finished in {time.time() - t1:.2f}s", flush=True)

        print("mesh extracted", flush=True)
        print("vertices :", len(mesh.vertices), flush=True)
        print("triangles:", len(mesh.triangles), flush=True)

        mesh_path = os.path.join(train_dir, name)
        ok = o3d.io.write_triangle_mesh(mesh_path, mesh)
        print("write ok:", ok, flush=True)
        print("mesh saved at", mesh_path, flush=True)

        print("starting post process ...", flush=True)
        t2 = time.time()
        mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        print(f"post process finished in {time.time() - t2:.2f}s", flush=True)

        post_path = os.path.join(train_dir, name.replace('.ply', '_post.ply'))
        ok2 = o3d.io.write_triangle_mesh(post_path, mesh_post)
        print("post write ok:", ok2, flush=True)
        print("post vertices :", len(mesh_post.vertices), flush=True)
        print("post triangles:", len(mesh_post.triangles), flush=True)
        print("mesh post processed saved at", post_path, flush=True)