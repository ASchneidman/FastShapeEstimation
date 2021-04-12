import torch
import numpy as np
import nvdiffrast.torch as dr
import math
from PIL import Image

import util


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


def fit_uv_mesh(initial_mesh: dict,
                target_dataset,
                max_iterations: int = 5000,
                resolution: int = 4,
                log_interval: int = 10,
                dispaly_interval = None,
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
    with np.load('cube_c.npz') as f:
        pos_idx, vtxp, col_idx, vtxc = f.values()
    fit_mesh({'pos_idx': torch.tensor(pos_idx.astype(np.int32)), 
    'vtx_pos': torch.tensor(vtxp.astype(np.float32)), 
    'col_idx': torch.tensor(col_idx.astype(np.int32)), 
    'vtx_col': torch.tensor(vtxc.astype(np.float32))}, None)