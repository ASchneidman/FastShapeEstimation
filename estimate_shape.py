'''
Demo code for the paper

Choy et al., 3D-R2N2: A Unified Approach for Single and Multi-view 3D Object
Reconstruction, ECCV 2016
'''
import os
import sys
import shutil
import numpy as np
from subprocess import call

import torch

from PIL import Image
from models import load_model
from lib.config import cfg, cfg_from_list
from lib.data_augmentation import preprocess_img
from lib.solver import Solver
from lib.voxel import voxel2obj

import skimage.measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


DEFAULT_WEIGHTS = 'output/ResidualGRUNet/default_model/checkpoint.pth'


def load_demo_images(paths):
    img_h = cfg.CONST.IMG_H
    img_w = cfg.CONST.IMG_W
    
    imgs = []
    
    for path in paths:
        img = Image.open(path)
        img = img.resize((img_h, img_w), Image.ANTIALIAS)
        img = preprocess_img(img, train=False)
        imgs.append([np.array(img).transpose( \
                        (2, 0, 1)).astype(np.float32)])
    ims_np = np.array(imgs).astype(np.float32)
    return torch.from_numpy(ims_np)


def main():
    '''Main demo function'''
    # Save prediction into a file named 'prediction.obj' or the given argument
    pred_file_name = sys.argv[1] if len(sys.argv) > 1 else 'prediction.obj'

    # load images
    demo_imgs = load_demo_images(paths)
    print(demo_imgs.shape)

    # Use the default network model
    NetClass = load_model('ResidualGRUNet')

    # Define a network and a solver. Solver provides a wrapper for the test function.
    net = NetClass()  # instantiate a network
    if torch.cuda.is_available():
        net.cuda()

    net.eval()

    solver = Solver(net)                # instantiate a solver
    solver.load(DEFAULT_WEIGHTS)

    # Run the network
    voxel_prediction, _ = solver.test_output(demo_imgs)
    voxel_prediction = voxel_prediction.detach().cpu().numpy()
    print(voxel_prediction.shape)
    # Save the prediction to an OBJ file (mesh file).
    voxel2obj(pred_file_name, voxel_prediction[0, 1] > cfg.TEST.VOXEL_THRESH)

    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(voxel_prediction[0, 1] > cfg.TEST.VOXEL_THRESH)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_xlim(0, 32)  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(0, 32)  # b = 10
    ax.set_zlim(0, 32)  # c = 16

    plt.tight_layout()
    plt.savefig('gen.png')


if __name__ == '__main__':
    # Set the batch size to 1
    # run: python estimate_shape.py
    # change the path to images and pretrained weights
    cfg_from_list(['CONST.BATCH_SIZE', 1])
    # weight: https://drive.google.com/open?id=1LtNhuUQdAeAyIUiuCavofBpjw26Ag6DP
    DEFAULT_WEIGHTS = 'path/to/weight.pkl'
    paths = ['images/chair.png']
    main(paths)
    
