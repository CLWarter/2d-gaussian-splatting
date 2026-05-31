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
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal

def material_band_colormap(x):
    """
    x: [1,H,W] in [0,1]
    returns [3,H,W] with 0.1 color bands
    """
    x = torch.clamp(x, 0.0, 1.0)
    q = torch.floor(x * 10.0 + 0.5) / 10.0

    out = torch.zeros((3, q.shape[1], q.shape[2]), device=q.device, dtype=q.dtype)

    v = q[0]

    def put(mask, r, g, b):
        out[0][mask] = r
        out[1][mask] = g
        out[2][mask] = b

    put(v < 0.05, 0.0, 0.0, 0.0)
    put((v >= 0.05) & (v < 0.15), 0.0, 0.0, 1.0)
    put((v >= 0.15) & (v < 0.25), 0.0, 1.0, 1.0)
    put((v >= 0.25) & (v < 0.35), 0.0, 1.0, 0.0)
    put((v >= 0.35) & (v < 0.45), 0.5, 1.0, 0.0)
    put((v >= 0.45) & (v < 0.55), 1.0, 1.0, 0.0)
    put((v >= 0.55) & (v < 0.65), 1.0, 0.5, 0.0)
    put((v >= 0.65) & (v < 0.75), 1.0, 0.25, 0.0)
    put((v >= 0.75) & (v < 0.85), 1.0, 0.0, 0.0)
    put((v >= 0.85) & (v < 0.95), 1.0, 0.0, 1.0)
    put(v >= 0.95, 1.0, 1.0, 1.0)

    return out

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, gt_image=None,
    collect_gaussian_luma=False,):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        # pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity    = pc.get_opacity
    ambient   = pc._ambient    # raw ambient parameter, per scene
    intensity = pc._intensity  # raw intensity parameter, per scene
    roughness = pc._roughness  # raw BRDF roughness parameter, per Gaussian
    metallic  = pc._metallic   # raw BRDF metallic parameter, per Gaussian

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    N = pc.get_xyz.shape[0]

    if collect_gaussian_luma and gt_image is not None:
        gt_luma = gt_image.detach().mean(dim=0, keepdim=True).contiguous()

        gauss_luma_sum = torch.zeros((N, 1), device="cuda", dtype=torch.float32)
        gauss_luma2_sum = torch.zeros((N, 1), device="cuda", dtype=torch.float32)
        gauss_luma_weight_sum = torch.zeros((N, 1), device="cuda", dtype=torch.float32)
    else:
        gt_luma = torch.empty(0, device="cuda", dtype=torch.float32)
        gauss_luma_sum = torch.empty(0, device="cuda", dtype=torch.float32)
        gauss_luma2_sum = torch.empty(0, device="cuda", dtype=torch.float32)
        gauss_luma_weight_sum = torch.empty(0, device="cuda", dtype=torch.float32)

    rendered_image, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        ambients = ambient,
        intensity = intensity,
        roughness = roughness,
        metallic = metallic,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        gt_luma=gt_luma,
        gauss_luma_sum=gauss_luma_sum,
        gauss_luma2_sum=gauss_luma2_sum,
        gauss_luma_weight_sum=gauss_luma_weight_sum
    )
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
    }

    if collect_gaussian_luma and gt_image is not None:
        rets["gauss_luma_sum"] = gauss_luma_sum
        rets["gauss_luma2_sum"] = gauss_luma2_sum
        rets["gauss_luma_weight_sum"] = gauss_luma_weight_sum

    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    render_metallic = None
    render_roughness = None

    if allmap.shape[0] > 7:
        render_metallic = torch.clamp(allmap[7:8], 0.0, 1.0)

    if allmap.shape[0] > 8:
        render_roughness = torch.clamp(allmap[8:9], 0.0, 1.0)

    # Mask invalid / transparent surface pixels for exported material maps.
    # These defaults are material-safe:
    #   metallic  = 0.0   dielectric
    #   roughness = 0.5   neutral roughness
    if render_metallic is not None or render_roughness is not None:
        material_valid = render_alpha > 0.5

        if render_metallic is not None:
            render_metallic = torch.where(
                material_valid,
                render_metallic,
                torch.zeros_like(render_metallic)
            )

        if render_roughness is not None:
            render_roughness = torch.where(
                material_valid,
                render_roughness,
                torch.full_like(render_roughness, 0.5)
            )

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()

    extra = {
        'rend_alpha': render_alpha,
        'rend_normal': render_normal,
        'rend_dist': render_dist,
        'surf_depth': surf_depth,
        'surf_normal': surf_normal,
    }

    if render_metallic is not None:
        extra["rend_metallic"] = render_metallic
        extra["rend_metallic_color"] = material_band_colormap(render_metallic)

    if render_roughness is not None:
        extra["rend_roughness"] = render_roughness
        extra["rend_roughness_color"] = material_band_colormap(render_roughness)

    rets.update(extra)

    return rets