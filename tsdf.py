import numpy as np
import open3d as o3d

print("Open3D version:", o3d.__version__, flush=True)

W, H = 64, 48

color = np.zeros((H, W, 3), dtype=np.uint8)
color[..., 0] = 255

depth = np.ones((H, W), dtype=np.float32) * 1.0

color_o3d = o3d.geometry.Image(color)
depth_o3d = o3d.geometry.Image(depth)

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_o3d,
    depth_o3d,
    depth_scale=1.0,
    depth_trunc=3.0,
    convert_rgb_to_intensity=False
)

intr = o3d.camera.PinholeCameraIntrinsic()
intr.set_intrinsics(W, H, 50.0, 50.0, W / 2.0, H / 2.0)

ext = np.eye(4, dtype=np.float64)

print("Creating UniformTSDFVolume...", flush=True)
vol = o3d.pipelines.integration.UniformTSDFVolume(
    length=4.0,
    resolution=64,
    sdf_trunc=0.04,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)

print("Before integrate...", flush=True)
vol.integrate(rgbd, intr, ext)
print("After integrate", flush=True)

mesh = vol.extract_triangle_mesh()
print("mesh verts:", len(mesh.vertices), flush=True)
print("mesh tris:", len(mesh.triangles), flush=True)