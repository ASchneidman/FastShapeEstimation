import torch
import numpy as np
import nvdiffrast.torch as dr
import math
from PIL import Image
import matplotlib.pyplot as plt

import util

from submodules.RNN.lib.voxel import write_obj


def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]


def render(glctx, mtx, pos, pos_idx, col_idx, vtx_col, resolution):
    pos_clip = transform_pos(mtx, pos)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])
    color, _    = dr.interpolate(vtx_col[None, ...], rast_out, col_idx)
    color       = dr.antialias(color, rast_out, pos_clip, pos_idx)
    color = color * torch.clamp(rast_out[..., -1:], 0, 1) # Mask out background.
    return color


def fit_mesh_col(
    initial_mesh: dict,
    target_dataset_dir: str,
    max_iterations: int = 10000,
    resolution: int = 256,
    log_interval: int = None,
    display_interval = None,
    display_res = 512,
    out_dir = None,
    mp4save_interval = None
    ):

    distance = 3

    target_dataset = util.ReferenceImages(target_dataset_dir, resolution, resolution)

    pos_idx = torch.from_numpy(initial_mesh['pos_idx'].astype(np.int32))
    vtx_pos = torch.from_numpy(initial_mesh['vtx_pos'].astype(np.float32))

    laplace = util.compute_laplace_matrix(vtx_pos, pos_idx).cuda()
    pos_idx = pos_idx.cuda()
    vtx_pos = vtx_pos.cuda()

    init_rot = util.rotate_z(-math.pi/2).cuda()
    vtx_pos = transform_pos(init_rot, vtx_pos)[0][:, 0:3]
    vtx_pos.requires_grad = True

    col_idx  = torch.from_numpy(initial_mesh['pos_idx'].astype(np.int32)).cuda()
    vtx_col  = torch.ones_like(vtx_pos) * 0.5
    vtx_col.requires_grad = True

    glctx = dr.RasterizeGLContext()


    M1 = torch.eye(len(target_dataset)).cuda()
    M1.requires_grad = True
    M2 = torch.eye(len(target_dataset)).cuda()
    M2.requires_grad = True

    #M3 = torch.zeros((3, vtx_pos.shape[0], len(target_dataset))).cuda()
    M3 = torch.zeros((3 * vtx_pos.shape[0], len(target_dataset))).cuda()
    M3.requires_grad = True

    lr_ramp = .1
    params = [{'params': [M1, M2, M3], 'lr': 1e-3}, {'params': vtx_col, 'lr': 1e-2}]
    #lambdas = [lambda x: max(0.01, 10**(-x*0.0005)), lambda x: lr_ramp**(float(x)/float(max_iterations))]


    optimizer    = torch.optim.Adam(params)
    #scheduler    = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)

    total_steps = 0


    for i in range(max_iterations):
        for j, (img, angle) in enumerate(target_dataset):
            img = img.cuda().permute(2,1,0)

            frame_tensor = torch.zeros(len(target_dataset))
            frame_tensor[j] = 1
            frame_tensor = frame_tensor.cuda()
            frame_tensor.requires_grad = True

            deltas = torch.matmul(M3, torch.matmul(M2, torch.matmul(M1, frame_tensor))).flatten()
            #deformed_vtxs = vtx_pos + deltas.T
            deformed_vtxs = (vtx_pos.flatten() + deltas).reshape((vtx_pos.shape[0], 3))

            # create the model-view-projection matrix
            # rotate model about z axis by angle
            #rot = util.rotate_y(angle)
            rot = torch.eye(4)
            # translate by distance
            tr = util.translate(z=-distance)
            # perspective projection
            proj = util.projection(x=0.4)

            mtx = proj.matmul(tr.matmul(rot)).cuda()
            mtx.requires_grad = True

            estimate = render(glctx, mtx, deformed_vtxs, pos_idx, col_idx, vtx_col, resolution)[0]

            # compute loss
            loss = torch.mean((estimate - img) ** 2)

            # compute regularizer
            reg = torch.mean((util.compute_curvature(deformed_vtxs, laplace) - util.compute_curvature(vtx_pos, laplace)) ** 2) + torch.mean(deltas**2)
            
            # combine
            loss = loss + reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()

            with torch.no_grad():
                #print(f"Loss: {loss}")
                # clamp color between 0 and 1
                vtx_col.clamp_(0, 1)

            if (display_interval and (i % display_interval == 0)) or (i == max_iterations - 1):
                print(loss)
                with torch.no_grad():
                    estimate = render(glctx, mtx, deformed_vtxs, pos_idx, col_idx, vtx_col, resolution)[0].detach().cpu().numpy()
                    Image.fromarray((estimate * 255).astype(np.uint8)).save('estimate.png')
                    img = img.detach().cpu().numpy()
                    Image.fromarray((img * 255).astype(np.uint8)).save('img.png')


    with torch.no_grad():
        for i, (im, _) in enumerate(target_dataset):
            frame_tensor = torch.zeros(len(target_dataset))
            frame_tensor[j] = 1
            frame_tensor = frame_tensor.cuda()

            deltas = torch.matmul(M3, torch.matmul(M2, torch.matmul(M1, frame_tensor))).flatten()
            deformed_vtxs = (vtx_pos.flatten() + deltas).reshape((vtx_pos.shape[0], 3))

            write_obj(f"frame_{i}.obj", deformed_vtxs.detach().cpu().tolist(), pos_idx.detach().cpu().tolist())

    np.savez('vtx_col.npz', vtx_col=vtx_col.cpu().detach().numpy())


if __name__ == '__main__':
    mesh = util.load_obj('sphere.obj')
    # mesh = util.load_obj('prediction.obj')
    vtx_pos = mesh['vtx_pos']
    # make all positive
    vtx_pos += vtx_pos.min()
    vtx_pos -= vtx_pos.min()
    vtx_pos /= vtx_pos.max()
    vtx_pos -= 0.5
    mesh['vtx_pos'] = vtx_pos

    for k,v in mesh.items():
        assert(v.shape[1] == 3)

    fit_mesh_col(mesh, 'images/bottle')