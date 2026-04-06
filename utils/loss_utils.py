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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def safe_normalize_map(n, eps=1e-6):
    # n: [3, H, W]
    denom = torch.linalg.norm(n, dim=0, keepdim=True).clamp_min(eps)
    return n / denom


def contribution_weighted_normal_terms(rend_normal, surf_normal, rend_alpha, image):
    """
    Returns:
        align_loss: alpha-weighted agreement loss between rend_normal and surf_normal
        smooth_loss: alpha-weighted, edge-aware smoothness loss on surf_normal
    """

    # Normalize defensively
    rend_normal = safe_normalize_map(rend_normal)
    surf_normal = safe_normalize_map(surf_normal)

    # Contribution/support weight.
    # Detach so this regularizer shapes normals primarily, instead of pushing alpha directly.
    w = rend_alpha.clamp(0.0, 1.0).pow(2.0).detach()

    # ---------- 1) contribution-weighted normal agreement ----------
    cos_sim = (rend_normal * surf_normal).sum(dim=0, keepdim=True).clamp(-1.0, 1.0)
    align_map = 1.0 - cos_sim

    w_sum = w.sum().clamp_min(1e-6)
    align_loss = (w * align_map).sum() / w_sum

    # ---------- 2) edge-aware local smoothness on surf_normal ----------
    # image: [3, H, W]
    img_dx = torch.mean(torch.abs(image[:, :, 1:] - image[:, :, :-1]), dim=0, keepdim=True)
    img_dy = torch.mean(torch.abs(image[:, 1:, :] - image[:, :-1, :]), dim=0, keepdim=True)

    n_dx = torch.mean(torch.abs(surf_normal[:, :, 1:] - surf_normal[:, :, :-1]), dim=0, keepdim=True)
    n_dy = torch.mean(torch.abs(surf_normal[:, 1:, :] - surf_normal[:, :-1, :]), dim=0, keepdim=True)

    # Edge-aware attenuation: strong image edges get less smoothing pressure
    edge_w_dx = torch.exp(-10.0 * img_dx)
    edge_w_dy = torch.exp(-10.0 * img_dy)

    w_dx = w[:, :, 1:]
    w_dy = w[:, 1:, :]

    smooth_x = (w_dx * edge_w_dx * n_dx).sum() / (w_dx.sum().clamp_min(1e-6))
    smooth_y = (w_dy * edge_w_dy * n_dy).sum() / (w_dy.sum().clamp_min(1e-6))
    smooth_loss = smooth_x + smooth_y

    return align_loss, smooth_loss

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

