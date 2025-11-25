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
import torch.nn as nn


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : i,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


class CoCoBlurKernel(nn.Module):
    
    res_pos = 3
    res_view = 10
    
    def __init__(self, opt, train_cams=None):
        super().__init__()

        self.train_cams = train_cams
        self.focal_planes = None
        
        self.num_coc_gaussians = opt.num_coc_gaussians
        self.width = opt.width
        self.num_hidden = opt.hidden
        
        self.embed_pos,  self.embed_pos_cnl  = get_embedder(self.res_pos, 3)
        self.embed_view, self.embed_view_cnl = get_embedder(self.res_view, 3)
        in_cnl = self.embed_pos_cnl + self.embed_view_cnl + 7
        
        hiddens = [nn.Linear(self.width, self.width) if i % 2 == 0 else nn.ReLU()
                    for i in range((self.num_hidden - 1) * 2)]

        self.linears = nn.Sequential(
                nn.Linear(in_cnl, self.width),
                nn.ReLU(),
                *hiddens,
        ).to("cuda")

        self.head_scale       = nn.Linear(self.width, 3 * (self.num_coc_gaussians + 1)).to("cuda")
        self.head_rotation    = nn.Linear(self.width, 4 * (self.num_coc_gaussians + 1)).to("cuda")
        self.head_pos         = nn.Linear(self.width, 3 *  self.num_coc_gaussians).to("cuda")

        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        # )
        # self.mlp_weight = nn.Conv2d(32, 1, kernel_size=1, bias=False)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.InstanceNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.InstanceNorm2d(64),
        )
        # self.mlp_weight = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, bias=False)
        self.mlp_weight = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        
            
    def forward(self, pos, scales, rotations, viewdirs, cam):

        if self.focal_planes is None:
            initial_depths = []
            focal_planes = []
            for c in self.train_cams:
                initial_depth = self.get_depth(pos, c.camera_center)
                initial_depths.append(initial_depth)
                focal_planes.append(initial_depth.mean())
            self.max_depth = torch.stack(initial_depths).max().item()
            print("MAX_DEPTH = ", self.max_depth)
            self.focal_planes = nn.Parameter(torch.stack(focal_planes).type(torch.float32) / self.max_depth, requires_grad=True)

        d_coc = self.get_coc(pos, cam)

        pos      = self.embed_pos(pos)
        viewdirs = self.embed_view(viewdirs)

        x = torch.cat([pos, viewdirs, scales, rotations], dim=-1)

        x = self.linears(x)

        scales_delta    = self.head_scale(x)
        rotations_delta = self.head_rotation(x)
        pos_delta       = self.head_pos(x)
        
        scales_delta    = torch.log(torch.clamp(d_coc * scales_delta + (torch.e - 0.01), min=torch.e, max=torch.e**1.1))
        rotations_delta = torch.log(torch.clamp(d_coc * rotations_delta + (torch.e - 0.01), min=torch.e, max=torch.e**1.1))
        pos_delta       = d_coc * pos_delta

        return {'scales_delta': scales_delta,
                'rotations_delta': rotations_delta,
                'pos_delta': pos_delta,
                'd_coc': d_coc}
    
    
    def get_depth(self, points, camera_center):
        return torch.sqrt(((points.detach() - camera_center) ** 2).sum(-1))
    
    
    def get_coc(self, points, cam):
        depth = self.get_depth(points, cam.camera_center)
        focal_plane = self.focal_planes[cam.uid] * self.max_depth
        d_coc = torch.abs(1 / depth - 1 / focal_plane)
        return d_coc[..., None]

    
    def get_weight(self, img):
        N, _, H, W = img.shape
        feat = self.conv(img)
        feat = feat[:, :, :H, :W]

        weight = self.mlp_weight(feat)
        weight = torch.softmax(weight, dim=0)

        return weight

        
