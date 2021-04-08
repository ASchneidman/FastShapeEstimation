import torch
import numpy as np
import nvdiffrast.torch as dr
import math

import util

def transform_pos(mtx, pos):
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, mtx.t())[None, ...]

def render(
    glctx, 
    vtx_pos,
    pos_idx,
    vtx_col,
    col_idx,
    angle: float,
    distance: float, 
    resolution
    ):

    # create the model-view-projection matrix
    # rotate model about z axis by angle
    rot = util.rotate(angle)
    # translate by distance
    tr = util.translate(z=-distance)
    # perspective projection
    proj = util.projection(x=0.4)

    mtx = proj.matmul(tr.matmul(rot)).cuda()


    clipped = transform_pos(mtx, vtx_pos)
    rast_out, _ = dr.rasterize(glctx, clipped, pos_idx, resolution=[resolution, resolution])
    color, _ = dr.interpolate(vtx_col[None, ...], rast_out, col_idx)
    color = dr.antialias(color, rast_out, clipped, pos_idx)

    return color

def fit_mesh(
    initial_mesh: dict,
    target_dataset,
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


    #optimizer    = torch.optim.Adam([vtx_pos, vtx_col_opt], lr=1e-2)
    #scheduler    = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.01, 10**(-x*0.0005)))

    total_steps = 0

    estimate = render(glctx, vtx_pos, pos_idx, vtx_col, col_idx, 0, 3.5, 256)
    img = estimate.detach().cpu().numpy()
    print(img.shape)
    while True:
        util.display_image(img[0], title='test', size=512)
    return

    for i in range(max_iterations):
        for angle, distance, img in target_dataset:
            img = img.cuda()

            estimate = render(glctx, vtx_pos, pos_idx, vtx_col, col_idx, angle, distance, resolution)

            # compute loss
            loss = torch.mean((estimate - img) ** 2)
            optimizer.zero_grad()
            loss.backward()
            scheduler.step()

            total_steps += 1


if __name__ == '__main__':
    with np.load('cube_c.npz') as f:
        pos_idx, vtxp, col_idx, vtxc = f.values()
    fit_mesh({'pos_idx': torch.tensor(pos_idx.astype(np.int32)), 
    'vtx_pos': torch.tensor(vtxp.astype(np.float32)), 
    'col_idx': torch.tensor(col_idx.astype(np.int32)), 
    'vtx_col': torch.tensor(vtxc.astype(np.float32))}, None)