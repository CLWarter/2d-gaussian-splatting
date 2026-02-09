import sys
from scene import Scene, GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render, network_gui
from utils.image_utils import render_net_image
import torch
import json, os

def load_lighting_cfg_for_model(model_path: str, override_path: str = ""):
    # Viewer rule (recommended):
    # 1) override_path if set
    # 2) else model_path/cfg_lighting.json if present
    if override_path:
        with open(override_path, "r", encoding="utf-8") as f:
            return json.load(f)
    candidate = os.path.join(model_path, "cfg_lighting.json")
    if os.path.isfile(candidate):
        with open(candidate, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def view(dataset, pipe, iteration):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    ambient = 0.2

    while True:
        with torch.no_grad():
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer, ambient)
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0]
                        # Add more metrics as needed
                    }
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                except Exception as e:
                    raise e
                    print('Viewer closed')
                    exit(0)

if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Exporting script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--iteration', type=int, default=30000)
    parser.add_argument('--lighting_cfg', type=str, default="",
                        help="Optional lighting cfg JSON override. If empty, uses <model_path>/cfg_lighting.json")
    args = parser.parse_args(sys.argv[1:])
    print("View: " + args.model_path)
    network_gui.init(args.ip, args.port)

    try:
        from diff_surfel_rasterization import _C
        from utils.lighting_config import pack_lighting_cfg

        cfg = {}

        if args.lighting_cfg:
            with open(args.lighting_cfg, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            cfg_path = os.path.join(args.model_path, "cfg_lighting.json")
            if os.path.isfile(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)

        _C.set_lighting_config(pack_lighting_cfg(cfg))

    except Exception as e:
        print(f"[viewer] lighting cfg upload failed: {e}")
    
    view(lp.extract(args), pp.extract(args), args.iteration)

    print("\nViewing complete.")