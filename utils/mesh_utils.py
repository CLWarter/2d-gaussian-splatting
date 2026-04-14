#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import numpy as np
import os
import math
from tqdm import tqdm
from utils.render_utils import save_img_f32, save_img_u8
from functools import partial
import open3d as o3d
import trimesh
import time

def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        print(f"[to_cam_open3d] start {i+1}/{len(viewpoint_stack)}", flush=True)

        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        print(f"[to_cam_open3d] size: {W} x {H}", flush=True)

        proj = viewpoint_cam.projection_matrix.detach().cpu().float()
        wv = viewpoint_cam.world_view_transform.detach().cpu().float()

        print(f"[to_cam_open3d] proj nan: {torch.isnan(proj).any().item()}", flush=True)
        print(f"[to_cam_open3d] proj inf: {torch.isinf(proj).any().item()}", flush=True)
        print(f"[to_cam_open3d] wv nan: {torch.isnan(wv).any().item()}", flush=True)
        print(f"[to_cam_open3d] wv inf: {torch.isinf(wv).any().item()}", flush=True)

        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W - 1) / 2],
            [0, H / 2, 0, (H - 1) / 2],
            [0, 0, 0, 1]
        ], dtype=torch.float32).T

        intrins = (proj @ ndc2pix)[:3, :3].T

        fx = float(intrins[0, 0].item())
        fy = float(intrins[1, 1].item())
        cx = float(intrins[0, 2].item())
        cy = float(intrins[1, 2].item())

        print(f"[to_cam_open3d] fx fy cx cy = {fx}, {fy}, {cx}, {cy}", flush=True)

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(W, H, fx, fy, cx, cy)

        extrinsic = np.ascontiguousarray(wv.T.numpy(), dtype=np.float64)

        camera_traj.append({
            "intrinsic": intrinsic,
            "extrinsic": extrinsic
        })

        print(f"[to_cam_open3d] done {i+1}/{len(viewpoint_stack)}", flush=True)

    return camera_traj


class GaussianExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background)
        self.clean()

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        self.alphamaps = []
        self.rgbmaps = []
        self.normals = []
        self.depth_normals = []
        self.viewpoint_stack = []

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            render_pkg = self.render(viewpoint_cam, self.gaussians)
            rgb = render_pkg['render']
            alpha = render_pkg['rend_alpha']
            normal = torch.nn.functional.normalize(render_pkg['rend_normal'], dim=0)
            depth = render_pkg['surf_depth']
            depth_normal = render_pkg['surf_normal']
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())
            self.alphamaps.append(alpha.cpu())
            self.normals.append(normal.cpu())
            self.depth_normals.append(depth_normal.cpu())
        
        # self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        # self.depthmaps = torch.stack(self.depthmaps, dim=0)
        # self.alphamaps = torch.stack(self.alphamaps, dim=0)
        # self.depth_normals = torch.stack(self.depth_normals, dim=0)
        self.estimate_bounding_sphere()

    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        from utils.render_utils import transform_poses_pca, focus_point_fn
        torch.cuda.empty_cache()
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        center = (focus_point_fn(poses))
        self.radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.
        
        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...", flush=True)
        print(f"voxel_size: {voxel_size}", flush=True)
        print(f"sdf_trunc: {sdf_trunc}", flush=True)
        print(f"depth_truc: {depth_trunc}", flush=True)

        print("creating TSDF volume ...", flush=True)
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        print("TSDF volume created", flush=True)

        cams_o3d = to_cam_open3d(self.viewpoint_stack)
        print(f"number of views: {len(cams_o3d)}", flush=True)

        for i, cam_o3d in enumerate(cams_o3d):
            
            t_view = time.time()
            print(f"[TSDF] entered loop for view {i+1}/{len(cams_o3d)}", flush=True)

            print(f"[TSDF] getting rgb for view {i+1}", flush=True)
            rgb = self.rgbmaps[i]
            print(f"[TSDF] got rgb for view {i+1}", flush=True)

            print(f"[TSDF] getting depth for view {i+1}", flush=True)
            depth = self.depthmaps[i]
            print(f"[TSDF] got depth for view {i+1}", flush=True)

            print(f"[TSDF] rgb shape: {tuple(rgb.shape)}", flush=True)
            print(f"[TSDF] depth shape: {tuple(depth.shape)}", flush=True)
            print(f"[TSDF] depth min/max: {depth.min().item()} / {depth.max().item()}", flush=True)
            print(f"[TSDF] depth has nan: {torch.isnan(depth).any().item()}", flush=True)
            print(f"[TSDF] depth has inf: {torch.isinf(depth).any().item()}", flush=True)

            if mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
                print(f"[TSDF] applying mask for view {i+1}", flush=True)
                depth[(self.viewpoint_stack[i].gt_alpha_mask < 0.5)] = 0
                print(f"[TSDF] mask applied for view {i+1}", flush=True)

            print(f"[TSDF] converting rgb to numpy for view {i+1}", flush=True)
            rgb_np = np.asarray(
                np.clip(rgb.permute(1, 2, 0).cpu().numpy(), 0.0, 1.0) * 255,
                order="C",
                dtype=np.uint8
            )
            print(f"[TSDF] rgb converted for view {i+1}", flush=True)

            print(f"[TSDF] converting depth to numpy for view {i+1}", flush=True)
            depth_np = np.asarray(depth.squeeze(0).cpu().numpy(), dtype=np.float32, order="C")
            print(f"[TSDF] depth converted for view {i+1}", flush=True)

            print(f"[TSDF] creating Open3D images for view {i+1}", flush=True)
            color_o3d = o3d.geometry.Image(rgb_np)
            depth_o3d = o3d.geometry.Image(depth_np)
            print(f"[TSDF] Open3D images created for view {i+1}", flush=True)

            print(f"[TSDF] creating RGBD for view {i+1}", flush=True)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
                depth_scale=1.0
            )
            print(f"[TSDF] RGBD created for view {i+1}", flush=True)

            print("[TSDF] extrinsic for view 1:", flush=True)
            print(cam_o3d["extrinsic"], flush=True)
            print("[TSDF] extrinsic det(upper3x3):",
                np.linalg.det(cam_o3d["extrinsic"][:3, :3]),
                flush=True)
            print("[TSDF] rgb_np shape/dtype:", rgb_np.shape, rgb_np.dtype, flush=True)
            print("[TSDF] depth_np shape/dtype:", depth_np.shape, depth_np.dtype, flush=True)

            print(f"[TSDF] integrating view {i+1}", flush=True)
            volume.integrate(rgbd, intrinsic=cam_o3d["intrinsic"], extrinsic=cam_o3d["extrinsic"])
            print(f"[TSDF] finished view {i+1} in {time.time() - t_view:.2f}s", flush=True)

        print("finished TSDF integration", flush=True)
        print("extracting triangle mesh ...", flush=True)
        mesh = volume.extract_triangle_mesh()
        print("triangle mesh extracted", flush=True)
        print(f"mesh verts: {len(mesh.vertices)}", flush=True)
        print(f"mesh tris: {len(mesh.triangles)}", flush=True)

        return mesh

    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=1024):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets. 
        return o3d.mesh
        """
        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))
        
        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2-mag) * (y/mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam):
            """
                compute per frame sdf
            """
            new_points = torch.cat([points, torch.ones_like(points[...,:1])], dim=-1) @ viewpoint_cam.full_proj_transform
            z = new_points[..., -1:]
            pix_coords = (new_points[..., :2] / new_points[..., -1:])
            mask_proj = ((pix_coords > -1. ) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(depthmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(-1, 1)
            sampled_rgb = torch.nn.functional.grid_sample(rgbmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3,-1).T
            sdf = (sampled_depth-z)
            return sdf, sampled_rgb, mask_proj

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            """
                Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1/(2-torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
                samples = inv_contraction(samples)
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:,0]) * (-1)
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:,0])
            for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
                sdf, rgb, mask_proj = compute_sdf_perframe(i, samples,
                    depthmap = self.depthmaps[i],
                    rgbmap = self.rgbmaps[i],
                    viewpoint_cam=self.viewpoint_stack[i],
                )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:,None] + rgb[mask_proj]) / wp[:,None]
                # update weight
                weights[mask_proj] = wp
            
            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = (self.radius * 2 / N)
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        from utils.mcube_utils import marching_cubes_with_contraction
        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R+0.01, 1.9)

        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )
        
        # coloring the mesh
        torch.cuda.empty_cache()
        mesh = mesh.as_open3d
        print("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(torch.tensor(np.asarray(mesh.vertices)).float().cuda(), inv_contraction=None, voxel_size=voxel_size, return_rgb=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh

    @torch.no_grad()
    def export_image(self, path):
        render_path = os.path.join(path, "renders")
        gts_path = os.path.join(path, "gt")
        vis_path = os.path.join(path, "vis")
        alpha_path  = os.path.join(path, "alpha")   # 🔹 new
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        os.makedirs(alpha_path, exist_ok=True)      # 🔹 new
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            gt = viewpoint_cam.original_image[0:3, :, :]
            save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            save_img_f32(self.depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))
            
            # 🔹 NEW: alpha export (single-channel float tiff, like depth)
            alpha = self.alphamaps[idx][0].cpu().numpy()   # [H, W], values in [0,1]
            save_img_f32(alpha,
            os.path.join(alpha_path, 'alpha_{0:05d}'.format(idx) + ".tiff"))
            
            save_img_u8(self.normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'normal_{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.depth_normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'depth_normal_{0:05d}'.format(idx) + ".png"))
