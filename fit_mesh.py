import torch
import numpy
import nvdiffrast.torch as dr

import util

def render(
    glctx, 
    vtx_pos,
    pos_idx,
    vtx_col,
    col_idx,
    angle: torch.Tensor, 
    distance: torch.Tensor, 
    resolution
    ):

    # create the model-view-projection matrix
    mtx = util.projection()

def fit_mesh(
    initial_mesh: dict,
    target_dataset: torch.Dataset,
    max_iterations: int = 5000,
    resolution: int = 4,
    log_interval: int = 10,
    dispaly_interval = None,
    display_res = 512,
    out_dir = None,
    mp4save_interval = None
    ):
    pos_idx = initial_mesh['pos_idx'].cuda()
    col_idx = initial_mesh['col_idx'].cuda()
    vtx_pos = initial_mesh['vtx_pos'].cuda()
    vtx_col = initial_mesh['vtx_col'].cuda()

    glctx = dr.RasterizeGLContext()




