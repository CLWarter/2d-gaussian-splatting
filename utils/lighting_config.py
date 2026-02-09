import torch

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

def pack_lighting_cfg(cfg: dict) -> torch.Tensor:
    merged = dict(_DEFAULTS)
    if cfg:
        merged.update(cfg)
    vals16 = [float(merged[k]) for k in _KEYS]
    assert len(vals16) == CFG_LEN
    return torch.tensor(vals16, dtype=torch.float32, device="cpu").contiguous()
