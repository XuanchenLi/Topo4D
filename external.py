
"""
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file found here:
# https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md
#
# For inquiries contact  george.drettakis@inria.fr

#######################################################################################################################
##### NOTE: CODE IN THIS FILE IS NOT INCLUDED IN THE OVERALL PROJECT'S MIT LICENSE #####
##### USE OF THIS CODE FOLLOWS THE COPYRIGHT NOTICE ABOVE #####
#######################################################################################################################
"""

import numpy as np
import torch
import torch.nn.functional as func
from torch.autograd import Variable
from math import exp


def build_rotation(q):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device='cuda')
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot

def build_quaterion(directions):
    # Normalize each direction vector to get unit vectors
    unit_directions = directions / torch.norm(directions, dim=1, keepdim=True)

    # Define the z-axis unit vector, repeated N times
    z_axis = torch.tensor([1.0, 0.0, 0.0]).repeat(directions.shape[0], 1)

    # Compute the rotation axes (cross product of z_axis and unit_directions)
    rotation_axes = torch.cross(z_axis, unit_directions)
    rotation_angles = torch.acos(torch.sum(z_axis * unit_directions, dim=1))

    # Compute quaternion components
    w = torch.cos(rotation_angles / 2)
    xyz = rotation_axes * torch.sin(rotation_angles / 2).unsqueeze(1)
    quaternions = torch.cat((w.unsqueeze(1), xyz), dim=1)

    return quaternions


def calc_mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def calc_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def calc_ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = func.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = func.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = func.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = func.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = func.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def accumulate_mean2d_gradient(variables):
    variables['means2D_gradient_accum'][variables['seen']] += torch.norm(
        variables['means2D'].grad[variables['seen'], :2], dim=-1)
    variables['denom'][variables['seen']] += 1
    return variables


def update_params_and_optimizer(new_params, params, optimizer):
    for k, v in new_params.items():
        group = [x for x in optimizer.param_groups if x["name"] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)

        stored_state["exp_avg"] = torch.zeros_like(v)
        stored_state["exp_avg_sq"] = torch.zeros_like(v)
        del optimizer.state[group['params'][0]]

        group["params"][0] = torch.nn.Parameter(v.requires_grad_(True))
        optimizer.state[group['params'][0]] = stored_state
        params[k] = group["params"][0]
    return params


def cat_params_to_optimizer(new_params, params, optimizer):
    for k, v in new_params.items():
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(v)), dim=0)
            stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(v)), dim=0)
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            params[k] = group["params"][0]
    return params


def remove_points(to_remove, params, variables, optimizer):
    to_keep = ~to_remove
    keys = [k for k in params.keys() if k not in ['cam_m', 'cam_c']]
    for k in keys:
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][to_keep]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_keep]
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter((group["params"][0][to_keep].requires_grad_(True)))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(group["params"][0][to_keep].requires_grad_(True))
            params[k] = group["params"][0]
    variables['means2D_gradient_accum'] = variables['means2D_gradient_accum'][to_keep]
    variables['denom'] = variables['denom'][to_keep]
    variables['max_2D_radius'] = variables['max_2D_radius'][to_keep]
    return params, variables


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def densify(params, variables, optimizer, i):
    if i <= 5000:
        variables = accumulate_mean2d_gradient(variables)
        grad_thresh = 0.0002
        if (i >= 500) and (i % 100 == 0):
            grads = variables['means2D_gradient_accum'] / variables['denom']
            grads[grads.isnan()] = 0.0
            to_clone = torch.logical_and(grads >= grad_thresh, (
                        torch.max(torch.exp(params['log_scales']), dim=1).values <= 0.01 * variables['scene_radius']))
            new_params = {k: v[to_clone] for k, v in params.items() if k not in ['cam_m', 'cam_c']}
            params = cat_params_to_optimizer(new_params, params, optimizer)
            num_pts = params['means3D'].shape[0]

            padded_grad = torch.zeros(num_pts, device="cuda")
            padded_grad[:grads.shape[0]] = grads
            to_split = torch.logical_and(padded_grad >= grad_thresh,
                                         torch.max(torch.exp(params['log_scales']), dim=1).values > 0.01 * variables[
                                             'scene_radius'])
            n = 2  # number to split into
            new_params = {k: v[to_split].repeat(n, 1) for k, v in params.items() if k not in ['cam_m', 'cam_c']}
            stds = torch.exp(params['log_scales'])[to_split].repeat(n, 1)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(params['unnorm_rotations'][to_split]).repeat(n, 1, 1)
            new_params['means3D'] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            new_params['log_scales'] = torch.log(torch.exp(new_params['log_scales']) / (0.8 * n))
            params = cat_params_to_optimizer(new_params, params, optimizer)
            num_pts = params['means3D'].shape[0]

            variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda")
            variables['denom'] = torch.zeros(num_pts, device="cuda")
            variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda")
            to_remove = torch.cat((to_split, torch.zeros(n * to_split.sum(), dtype=torch.bool, device="cuda")))
            params, variables = remove_points(to_remove, params, variables, optimizer)

            remove_threshold = 0.25 if i == 5000 else 0.005
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()
            if i >= 3000:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)
            params, variables = remove_points(to_remove, params, variables, optimizer)

            torch.cuda.empty_cache()

        if i > 0 and i % 3000 == 0:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def dense_remove_points(to_remove, params, variables, optimizer):
    to_keep = ~to_remove
    keys = [k for k in params.keys() if k.startswith("dense_")]
    for k in keys:
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][to_keep]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_keep]
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter((group["params"][0][to_keep].requires_grad_(True)))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(group["params"][0][to_keep].requires_grad_(True))
            params[k] = group["params"][0]
    variables['dense_means2D_gradient_accum'] = variables['dense_means2D_gradient_accum'][to_keep]
    variables['dense_denom'] = variables['dense_denom'][to_keep]
    variables['dense_max_2D_radius'] = variables['dense_max_2D_radius'][to_keep]
    return params, variables




def dense_densify(params, variables, optimizer, grad_thresh, min_opacity, extent, max_screen_size, timestep, iteration):
    seen = variables['dense_seen']
    is_initial_timestep = (timestep == 0)
    # Densification
    if iteration < 45000 and is_initial_timestep:
        # Keep track of max radii in image-space for pruning
        variables['dense_means2D_gradient_accum'][seen] += torch.norm(variables['dense_means2D'].grad[seen,:2], dim=-1)
        variables['dense_denom'][seen] += 1
        
        if iteration > 500 and iteration % 100 == 0:
            params, variables = dense_densify_aux(params, variables, optimizer, grad_thresh, min_opacity, extent, max_screen_size=max_screen_size, iteration=iteration)
        
        if iteration > 0 and iteration % 3000 == 0:
            new_params = {'dense_logit_opacities': inverse_sigmoid(torch.min(torch.sigmoid(params['dense_logit_opacities']), torch.ones_like(params['dense_logit_opacities'])*0.01))}
            params = update_params_and_optimizer(new_params, params, optimizer)
            
        # if iteration % 3000 == 0:
        #     opacities_new = inverse_sigmoid(torch.min(torch.sigmoid(params['dense_logit_opacities']), torch.ones_like(params['dense_logit_opacities'])*0.01))
        #     params['dense_logit_opacities'] = opacities_new
            # optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
            # self._opacity = optimizable_tensors["opacity"]

    return params, variables


def dense_densify_aux(params, variables, optimizer, grad_thresh, min_opacity, extent, max_screen_size, iteration):
    grads = variables['dense_means2D_gradient_accum'] / variables['dense_denom']
    grads[grads.isnan()] = 0.0
    to_clone = torch.logical_and(grads >= grad_thresh, (
                torch.max(torch.exp(params['dense_log_scales']), dim=1).values <= 0.0008 * extent))
    new_params = {k: v[to_clone] for k, v in params.items() if k.startswith("dense_")}
    params = cat_params_to_optimizer(new_params, params, optimizer)
    num_pts = params['dense_means3D'].shape[0]

    padded_grad = torch.zeros(num_pts, device="cuda")
    padded_grad[:grads.shape[0]] = grads
    to_split = torch.logical_and(padded_grad >= grad_thresh,
                                 torch.max(torch.exp(params['dense_log_scales']), dim=1).values > 0.0008 * extent)
    
    n = 2  # number to split into
    new_params = {k: v[to_split].repeat((n,) + (1,) * (v[to_split].dim() - 1)) for k, v in params.items() if k.startswith("dense_")}

    stds = torch.exp(params['dense_log_scales'])[to_split].repeat(n, 1)
    means = torch.zeros((stds.shape[0], 3), device="cuda")
    samples = torch.normal(mean=means, std=stds)
    rots = build_rotation(params['dense_unnorm_rotations'][to_split]).repeat(n, 1, 1)
    new_params['dense_means3D'] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
    new_params['dense_log_scales'] = torch.log(torch.exp(new_params['dense_log_scales']) / (0.8 * (n + 1)))
    # for k, v in new_params.items():
    #     if k != 'dense_means3D':
    #         params[k][to_split] = v[0]
    params = cat_params_to_optimizer(new_params, params, optimizer)
    num_pts = params['dense_means3D'].shape[0]

    variables['dense_means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda")
    variables['dense_denom'] = torch.zeros(num_pts, device="cuda")
    variables['dense_max_2D_radius'] = torch.zeros(num_pts, device="cuda")
    
    to_remove = torch.cat((to_split, torch.zeros(n * to_split.sum(), dtype=torch.bool, device="cuda")))
    #to_remove[:params["means3D"].shape[0]] = False
    params, variables = dense_remove_points(to_remove, params, variables, optimizer)

    to_remove = (torch.sigmoid(params['dense_logit_opacities']) < min_opacity).squeeze()
    if max_screen_size is not None:
        big_points_vs =  variables['dense_max_2D_radius'] > max_screen_size
        big_points_ws = torch.exp(params['dense_log_scales']).max(dim=1).values > 0.1 * extent
        to_remove = torch.logical_or(to_remove, big_points_ws)
        to_remove = torch.logical_or(to_remove, big_points_vs)
    #to_remove[:params["means3D"].shape[0]] = False
    params, variables = dense_remove_points(to_remove, params, variables, optimizer)
    print("Vertex number: %d " % params['dense_means3D'].shape[0])
    torch.cuda.empty_cache()
    return params, variables


def calculate_q_from_n(normals, default_direction=np.array([0, 0, 1])):
    # 确保法线是单位向量
    normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

    # 计算旋转轴
    rotation_axes = np.cross(default_direction, normals)

    # 计算旋转角度
    angles = np.arccos(np.clip(np.dot(normals, default_direction), -1.0, 1.0))

    # 计算四元数的各个分量
    sin_angles_2 = np.sin(angles / 2)
    quaternions = np.zeros((normals.shape[0], 4))
    quaternions[:, 0] = np.cos(angles / 2)  # qw
    quaternions[:, 1:4] = rotation_axes * sin_angles_2[:, np.newaxis]  # qx, qy, qz

    return quaternions