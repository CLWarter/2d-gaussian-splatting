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
    K = get_intrinsics(cam)
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
    pts_h = torch.cat([pts_world.view(-1,3), torch.ones((pts_world.numel()//3,1), device="cuda")], dim=1)
    pts_view = pts_h @ cam.world_view_transform.T
    z = pts_view[:, 2:3]

    K = get_intrinsics(cam)
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