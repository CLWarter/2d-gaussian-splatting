import torch
import torch.nn.functional as F
from utils.graphics_utils import fov2focal

def get_intrinsics(cam):
    W, H = cam.image_width, cam.image_height
    fx = fov2focal(cam.FoVx, W)
    fy = fov2focal(cam.FoVy, H)
    cx = W * 0.5
    cy = H * 0.5
    K = torch.tensor([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32, device="cuda")
    return K

def depth_to_world_points(cam, depth):
    H, W = depth.shape[-2:]
    K = get_intrinsics_scaled(cam, H, W)
    Kinv = torch.inverse(K)

    ys, xs = torch.meshgrid(
        torch.arange(H, device="cuda", dtype=torch.float32),
        torch.arange(W, device="cuda", dtype=torch.float32),
        indexing="ij"
    )

    pix = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).view(-1, 3)
    rays_cam = pix @ Kinv.T
    z = depth.view(-1, 1)

    pts_cam = rays_cam * z

    c2w = torch.inverse(cam.world_view_transform.T)
    pts_cam_h = torch.cat([pts_cam, torch.ones_like(z)], dim=1)
    pts_world = pts_cam_h @ c2w
    return pts_world[:, :3].view(H, W, 3)

def world_to_view_depth_uv(cam, pts_world):
    H, W = pts_world.shape[:2]

    pts_h = torch.cat([
        pts_world.view(-1, 3),
        torch.ones((pts_world.numel() // 3, 1), device="cuda")
    ], dim=1)

    pts_view = pts_h @ cam.world_view_transform.T
    z = pts_view[:, 2:3]

    K = get_intrinsics_scaled(cam, H, W)
    pix = pts_view[:, :3] @ K.T
    uv = pix[:, :2] / (pix[:, 2:3] + 1e-8)

    u = uv[:, 0]
    v = uv[:, 1]

    u_norm = 2.0 * (u / max(W - 1, 1)) - 1.0
    v_norm = 2.0 * (v / max(H - 1, 1)) - 1.0
    grid = torch.stack([u_norm, v_norm], dim=-1)

    in_bounds = (
        (u_norm >= -1.0) & (u_norm <= 1.0) &
        (v_norm >= -1.0) & (v_norm <= 1.0) &
        (z[:, 0] > 0.0)
    )

    return grid.view(H, W, 2), z.view(H, W, 1), in_bounds.view(H, W, 1)

def sample_map(map_chw, grid_hw2):
    grid = grid_hw2.unsqueeze(0)
    sampled = F.grid_sample(
        map_chw.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )
    return sampled.squeeze(0)

def normals_to_world(cam, normal_chw):
    R = torch.inverse(cam.world_view_transform.T)[:3, :3]
    n = normal_chw.permute(1,2,0) @ R[:3,:3].T
    n = torch.nn.functional.normalize(n, dim=-1)
    return n.permute(2,0,1)

def get_intrinsics_scaled(cam, H, W):
    fx = fov2focal(cam.FoVx, cam.image_width) * (W / cam.image_width)
    fy = fov2focal(cam.FoVy, cam.image_height) * (H / cam.image_height)
    cx = W * 0.5
    cy = H * 0.5
    K = torch.tensor([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32, device="cuda")
    return K

def compute_mv_losses(
    viewpoint_cam_a,
    out_a,
    viewpoint_cam_b,
    out_b,
    max_samples=5000,
    use_normal=True,
    use_abs_normal=True,
):
    depth_a_mv = F.interpolate(
        out_a["surf_depth"].unsqueeze(0),
        scale_factor=0.25,
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    depth_b_mv = F.interpolate(
        out_b["surf_depth"].unsqueeze(0),
        scale_factor=0.25,
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    alpha_a_mv = F.interpolate(
        out_a["rend_alpha"].unsqueeze(0),
        scale_factor=0.25,
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    alpha_b_mv = F.interpolate(
        out_b["rend_alpha"].unsqueeze(0),
        scale_factor=0.25,
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    pts_world = depth_to_world_points(viewpoint_cam_a, depth_a_mv)
    grid_b, z_b_pred, in_bounds = world_to_view_depth_uv(viewpoint_cam_b, pts_world)

    depth_b_samp = sample_map(depth_b_mv, grid_b)
    alpha_b_samp = sample_map(alpha_b_mv, grid_b)

    w = (alpha_a_mv * alpha_b_samp).detach() * in_bounds.permute(2, 0, 1).float()

    flat_w = w.view(-1)
    valid_idx = (flat_w > 0.0).nonzero(as_tuple=False).squeeze(1)

    mv_depth_loss = torch.tensor(0.0, device="cuda")
    mv_normal_loss = torch.tensor(0.0, device="cuda")

    if valid_idx.numel() > 0:
        if valid_idx.numel() > max_samples:
            perm = torch.randperm(valid_idx.numel(), device="cuda")[:max_samples]
            valid_idx = valid_idx[perm]

        z_pred_flat = z_b_pred.permute(2, 0, 1).contiguous().view(-1)
        depth_b_flat = depth_b_samp.view(-1)

        depth_res = (depth_b_flat[valid_idx] - z_pred_flat[valid_idx]) / (z_pred_flat[valid_idx].abs() + 1e-3)
        w_sel = flat_w[valid_idx]

        mv_depth_loss = (w_sel * torch.sqrt(depth_res * depth_res + 1e-6)).sum() / (w_sel.sum() + 1e-8)

        if use_normal:
            normal_a_mv = F.interpolate(
                out_a["rend_normal"].unsqueeze(0),
                scale_factor=0.25,
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

            normal_b_mv = F.interpolate(
                out_b["rend_normal"].unsqueeze(0),
                scale_factor=0.25,
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

            normal_b_samp = sample_map(normal_b_mv, grid_b)

            normal_a_w = normals_to_world(viewpoint_cam_a, normal_a_mv)
            normal_b_w = normals_to_world(viewpoint_cam_b, normal_b_samp)

            dot_nb = (normal_a_w * normal_b_w).sum(dim=0, keepdim=True).view(-1)[valid_idx]
            if use_abs_normal:
                dot_nb = dot_nb.abs()

            mv_normal_loss = (w_sel * (1.0 - dot_nb)).sum() / (w_sel.sum() + 1e-8)

    return mv_depth_loss, mv_normal_loss