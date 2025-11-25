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
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import torch.nn.functional as F


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier=1.0, blurring=0, iteration=None, start_weight=None):
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
        image_height    = int(viewpoint_camera.image_height),
        image_width     = int(viewpoint_camera.image_width),
        tanfovx         = tanfovx,
        tanfovy         = tanfovy,
        bg              = bg_color,
        scale_modifier  = scaling_modifier,
        viewmatrix      = viewpoint_camera.world_view_transform,
        projmatrix      = viewpoint_camera.full_proj_transform,
        sh_degree       = pc.active_sh_degree,
        campos          = viewpoint_camera.camera_center,
        prefiltered     = False,
        debug           = pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else: 
        if not blurring:
            scales, rotations, shs = pc.get_scaling, pc.get_rotation, pc.get_features
            colors_precomp = None
            rendered_image, radii = rasterizer(means3D          = means3D,
                                               means2D          = means2D,
                                               shs              = shs,
                                               colors_precomp   = colors_precomp,
                                               opacities        = opacity,
                                               scales           = scales,
                                               rotations        = rotations,
                                               cov3D_precomp    = cov3D_precomp)
            
            return {"render": rendered_image,
                    "viewspace_points": screenspace_points,
                    "visibility_filter" : radii > 0,
                    "radii": radii}

        else:
            scales, rotations, shs = pc.get_scaling, pc.get_rotation, pc.get_features
            colors_precomp = None
            num_coc_gaussians = pc.CoCoBlurKernel.num_coc_gaussians

            coc_pkg = pc.CoCoBlurKernel(pos         = means3D.detach(),
                                        scales      = scales.detach(),
                                        rotations   = rotations.detach(),
                                        viewdirs    = viewpoint_camera.camera_center.repeat(means3D.shape[0], 1),
                                        cam         = viewpoint_camera)
            
            scales_delta    = coc_pkg['scales_delta'].view(-1, 3, num_coc_gaussians + 1)
            rotations_delta = coc_pkg['rotations_delta'].view(-1, 4, num_coc_gaussians + 1)
            pos_delta       = coc_pkg['pos_delta'].view(-1, 3, num_coc_gaussians)

            base_pos        = means3D
            base_scales     = scales * scales_delta[..., -1]
            base_rotations  = rotations * rotations_delta[..., -1]

            rendered_image, _radii = rasterizer(means3D         = base_pos,
                                                means2D         = means2D,
                                                shs             = shs,
                                                colors_precomp  = colors_precomp,
                                                opacities       = opacity,
                                                scales          = base_scales,
                                                rotations       = base_rotations,
                                                cov3D_precomp   = cov3D_precomp)

            renders = [rendered_image]
            viewspace_points = [screenspace_points]
            visibility_filter = [_radii > 0]
            radii = [_radii]

            for i in range(num_coc_gaussians):
                screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
                try:
                    screenspace_points.retain_grad()
                except:
                    pass
                
                means2D         = screenspace_points
                coc_pos         = means3D + pos_delta[..., i]
                coc_scales      = scales * scales_delta[..., i]
                coc_rotations   = rotations * rotations_delta[..., i]

                rendered_image, _radii = rasterizer(means3D = coc_pos,
                                                    means2D = means2D,
                                                    shs = shs,
                                                    colors_precomp = colors_precomp,
                                                    opacities = opacity,
                                                    scales = coc_scales,
                                                    rotations = coc_rotations,
                                                    cov3D_precomp = cov3D_precomp)

                renders.append(rendered_image)
                viewspace_points.append(screenspace_points)
                visibility_filter.append(_radii > 0)
                radii.append(_radii)

            if iteration is not None and iteration < start_weight:
                render = sum(renders) / len(renders)
            else:
                renders = torch.stack(renders)
                weight = pc.CoCoBlurKernel.get_weight(renders.detach())
                render = torch.sum(renders * weight, dim=0)

            return {"render": render,
                    "viewspace_points": viewspace_points,
                    "visibility_filter" : visibility_filter,
                    "radii": radii,}

