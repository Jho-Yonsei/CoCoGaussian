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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from PIL import Image
import numpy as np
from utils.image_utils import psnr
from metrics import compute_img_metric
import imageio
from scene.cameras import Camera
from scipy.spatial.transform import Rotation, Slerp

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def get_video_cams(views):
    cam = views[0]
    views_video = []

    for idx in range(len(views) - 1):
        Rs, Ts = interpolate_SE(views[idx].R, views[idx].T, views[idx + 1].R, views[idx + 1].T)
        for k in range(len(Rs)):
            view = Camera(colmap_id=cam.colmap_id, R=Rs[k], T=Ts[k], 
                            FoVx=cam.FoVx, FoVy=cam.FoVy, 
                            image=cam.original_image, gt_alpha_mask=None,
                            image_name=cam.image_name, uid=cam.uid, data_device=cam.data_device)
            views_video.append(view)

    return views_video
    

def interpolate_SE(R1, T1, R2, T2, num_points=40):

    slerp = Slerp([0, 1], Rotation.from_matrix([R1, R2]))

    interp_points = np.linspace(0, 1, num_points)

    Rs, Ts = [], []
    for alpha in interp_points:
        t = (1 - alpha) * T1 + alpha * T2
        R_interp = slerp(alpha).as_matrix()
        Rs.append(R_interp)
        Ts.append(t)
    return Rs, Ts



def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")

    makedirs(render_path, exist_ok=True)


    rendering_range_1 = list(range(0, 101))
    rendering_range_2 = list(range(99, -1, -1))
    rendering_range = rendering_range_1 + rendering_range_2

    views_video = get_video_cams(views)
    

    with torch.no_grad():
        render_pkgs = []
        for idx, view in enumerate(tqdm(views_video, desc="Rendering video progress")):
            render_pkgs.append(render(view, gaussians, pipeline, background, deblur=0)['render'])

        rgbs = [to8b(render_pkg.permute(1, 2, 0).cpu().numpy()) for render_pkg in render_pkgs]
        rgbs = np.stack(rgbs, axis=0)

    imageio.mimwrite("./visualize_video/caps.mp4", rgbs, fps=60, quality=8)

    

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, 1)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
