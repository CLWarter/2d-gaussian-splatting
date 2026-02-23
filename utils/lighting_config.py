import json
import os
from typing import Dict, Any, Optional
import torch
import copy

CFG_LEN = 16

_DEFAULTS = {
    "enable_fwd": 1,
    "enable_bwd": 1,
    "light_mode": 3,
    "ambient_mode": 2,
    "ambient_fixed": 0.02,
    "lambert_mode": 0,
    "phong_ks_mode": 1,
    "phong_shiny_mode": 0,
    "phong_ks": 0.1,
    "phong_shininess": 8.0,
    "spec_gating": 2,
    "energy_comp": 1,
    "use_spot": 1,
    "spot_inner_deg": 15.0,
    "spot_outer_deg": 35.0,
    "spot_exp": 1.5,
}

_KEYS = [
    "enable_fwd", "enable_bwd", "light_mode", "ambient_mode", "ambient_fixed",
    "lambert_mode", "phong_ks_mode", "phong_shiny_mode", "phong_ks", "phong_shininess",
    "spec_gating", "energy_comp", "use_spot", "spot_inner_deg", "spot_outer_deg", "spot_exp"
]

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_cfg(cfg: dict) -> dict:
    out = copy.deepcopy(_DEFAULTS)
    if cfg:
        out.update(cfg)
    return out

def pack_lighting_cfg(cfg: dict, device="cpu") -> torch.Tensor:
    c = normalize_cfg(cfg)
    vals16 = [float(c[k]) for k in _KEYS]
    return torch.tensor(vals16, dtype=torch.float32, device=device).contiguous()

def resolve_cfg_path(model_path: str, override_path: str) -> str:
    if override_path:
        return override_path
    return os.path.join(model_path, "cfg_lighting.json")

def save_cfg(model_path: str, cfg: dict):
    os.makedirs(model_path, exist_ok=True)
    path = os.path.join(model_path, "cfg_lighting.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(normalize_cfg(cfg), f, indent=2, sort_keys=True)
    return path