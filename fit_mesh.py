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


def render(glctx, mtx, pos, pos_idx, uv, uv_idx, tex, resolution, enable_mip, max_mip_level):
    pos_clip = transform_pos(mtx, pos)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])

    if enable_mip:
        texc, texd = dr.interpolate(uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
        color = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    else:
        texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
        color = dr.texture(tex[None, ...], texc, filter_mode='linear')

    color = color * torch.clamp(rast_out[..., -1:], 0, 1) # Mask out background.
    return color


def init_uv(stack_count=128, sector_count=128):
        sector_step = 2 * np.pi / sector_count
        stack_step = np.pi / stack_count
        uv_vertices = []
        uv_idx = []

        for i in range(stack_count + 1):
            for j in range(sector_count + 1):
                s = j / sector_count #u
                t = i / stack_count  #v
                uv_vertices.append((s, t))

        for i in range(stack_count):
            vi1 = i * (sector_count + 1)
            vi2 = (i + 1) * (sector_count + 1)
            for j in range(sector_count):
                v1, v2 = uv_vertices[vi1], uv_vertices[vi2]
                v3, v4 = uv_vertices[vi1 + 1], uv_vertices[vi2 + 1]
                
                # first or last stack triangle, others quad
                if i == 0:
                    uv_idx.append([vi1, vi2, vi2 + 1])
                elif i == stack_count - 1:
                    uv_idx.append([vi1, vi2, vi1 + 1])
                else:
                    uv_idx.append([vi1, vi2, vi1 + 1])
                    uv_idx.append([vi1 + 1, vi2, vi2 + 1])
                vi1 += 1
                vi2 += 1

        uv_vertices = np.array(uv_vertices, dtype=np.float32)
        uv_idx = np.array(uv_idx, dtype=np.int32)
        return uv_vertices, uv_idx


def fit_mesh(
    initial_mesh: dict,
    target_dataset_dir: str,
    max_iterations: int = 10000,
    resolution: int = 256,
    log_interval: int = 1000,
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

    uv, uv_idx = init_uv()
    uv_idx = uv_idx[:pos_idx.shape[0]]
    uv_idx  = torch.from_numpy(uv_idx.astype(np.int32)).cuda()
    vtx_uv  = torch.from_numpy(uv.astype(np.float32)).cuda()
    vtx_uv.requires_grad = True


    #col_idx  = torch.from_numpy(initial_mesh['col_idx'].astype(np.int32)).cuda()
    #vtx_col  = initial_mesh['vtx_col'].cuda()
    tex = torch.ones((1024, 1024, 3)).float() / 2
    tex = tex.cuda()
    tex.requires_grad = True

    glctx = dr.RasterizeGLContext()


    M1 = torch.eye(len(target_dataset)).cuda()
    M1.requires_grad = True
    M2 = torch.eye(len(target_dataset)).cuda()
    M2.requires_grad = True

    #M3 = torch.zeros((3, vtx_pos.shape[0], len(target_dataset))).cuda()
    M3 = torch.zeros((3 * vtx_pos.shape[0], len(target_dataset))).cuda()
    M3.requires_grad = True

    lr_ramp = .1
    params = [{'params': [M1, M2, M3], 'lr': 1e-3}, {'params': tex, 'lr': 1e-2}]
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
            
            estimate = render(glctx, mtx, deformed_vtxs, pos_idx, vtx_uv, uv_idx, tex, resolution, enable_mip=False, max_mip_level=4)[0]

            # compute loss
            loss = torch.mean((estimate - img) ** 2)

            # compute regularizer
            reg = torch.mean((util.compute_curvature(deformed_vtxs, laplace) - util.compute_curvature(vtx_pos, laplace)) ** 2)
            
            # combine
            loss = 5 * loss + 0 * reg


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()

            with torch.no_grad():
                # clamp texture between 0 and 1
                tex.clamp_(0, 1)

            if (display_interval and (i % display_interval == 0)) or (i == max_iterations - 1):
                with torch.no_grad():
                    estimate = render(glctx, mtx, deformed_vtxs, pos_idx, vtx_uv, uv_idx, tex, resolution, enable_mip=True, max_mip_level=4)[0].detach().cpu().numpy()
                    plt.imshow(estimate)
                    plt.show()
                    plt.imshow(img.detach().cpu().numpy())
                    plt.show()

            if log_interval and i % log_interval == 0:
                print(f"Loss: {loss}")
                print(M1.grad)

    with torch.no_grad():
        for i, (im, _) in enumerate(target_dataset):
            frame_tensor = torch.zeros(len(target_dataset))
            frame_tensor[j] = 1
            frame_tensor = frame_tensor.cuda()

            deltas = torch.matmul(M3, torch.matmul(M2, torch.matmul(M1, frame_tensor))).flatten()
            deformed_vtxs = (vtx_pos.flatten() + deltas).reshape((vtx_pos.shape[0], 3))

            write_obj(f"frame_{i}.obj", deformed_vtxs.detach().cpu().tolist(), pos_idx.detach().cpu().tolist())
    Image.fromarray((tex.detach().cpu().numpy() * 255).astype(np.uint8)).save('diff_render_tex.png')
    print("Outputted texture to diff_render_tex.png")
    



def fit_uv_mesh(initial_mesh: dict,
                target_dataset,
                max_iterations: int = 5000,
                resolution: int = 4,
                log_interval: int = 10,
                dispaly_interval = 1000,
                display_res = 512,
                out_dir = None,
                mp4save_interval = None
                ):
    glctx = dr.RasterizeGLContext()

    r_rot = util.random_rotation_translation(0.25)

    # Smooth rotation for display.
    ang = 0.0
    a_rot = np.matmul(util.rotate_x(-0.4), util.rotate_y(ang))
    dist = 2

    # Modelview and modelview + projection matrices.
    proj  = util.projection(x=0.4, n=1.0, f=200.0)
    r_mv  = np.matmul(util.translate(0, 0, -1.5-dist), r_rot)
    r_mvp = np.matmul(proj, r_mv).astype(np.float32)
    a_mv  = np.matmul(util.translate(0, 0, -3.5), a_rot)
    a_mvp = np.matmul(proj, a_mv).astype(np.float32)

    pos_idx = initial_mesh['pos_idx'].cuda()
    vtx_pos = initial_mesh['vtx_pos'].cuda()
    tex = np.ones((1024, 1024, 3), dtype=np.float32) / 2

    uv, uv_idx = init_uv()
    uv_idx = uv_idx[:pos_idx.shape[0]]
    pos_idx = torch.from_numpy(pos_idx.astype(np.int32)).cuda()
    vtx_pos = torch.from_numpy(pos.astype(np.float32)).cuda()
    uv_idx  = torch.from_numpy(uv_idx.astype(np.int32)).cuda()
    vtx_uv  = torch.from_numpy(uv.astype(np.float32)).cuda()
    tex     = torch.from_numpy(tex.astype(np.float32)).cuda()

    # Render reference and optimized frames. Always enable mipmapping for reference.
    color = render(glctx, r_mvp, vtx_pos, pos_idx, vtx_uv, uv_idx, tex, 1024, False, 0)
    Image.fromarray((color[0].detach().cpu().numpy() * 255).astype(np.uint8)).save('test.png')


if __name__ == '__main__':
    mesh = util.load_obj('sphere.obj')
    #mesh = util.load_obj('prediction.obj')
    vtx_pos = mesh['vtx_pos']
    # make all positive
    vtx_pos += vtx_pos.min()
    vtx_pos -= vtx_pos.min()
    vtx_pos /= vtx_pos.max()
    vtx_pos -= 0.5
    mesh['vtx_pos'] = vtx_pos

    for k,v in mesh.items():
        assert(v.shape[1] == 3)

    fit_mesh(mesh, 'images/bottle')