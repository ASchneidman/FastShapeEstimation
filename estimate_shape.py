import json
import numpy as np
import os
import torch
import torch.backends.cudnn
import torch.utils.data

import submodules.Pix2Vox.utils.binvox_visualization as binvox_visualization
import submodules.Pix2Vox.utils.data_transforms as data_transforms
import submodules.Pix2Vox.utils.network_utils as network_utils
import submodules.Pix2Vox.utils.data_loaders as data_loaders

from argparse import ArgumentParser
from datetime import datetime as dt

from submodules.Pix2Vox.models.encoder import Encoder
from submodules.Pix2Vox.models.decoder import Decoder
from submodules.Pix2Vox.models.refiner import Refiner
from submodules.Pix2Vox.models.merger import Merger
from submodules.Pix2Vox.config import cfg

from PIL import Image
import skimage.measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import fit_mesh


def infer(cfg, img, epoch_idx=-1):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
    CROP_SIZE = IMG_SIZE
    test_transforms = data_transforms.Compose([
        data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
        data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
        data_transforms.ToTensor(),
    ])

    inp = test_transforms(img)
    inp = inp.unsqueeze(0)

    # Set up networks
    encoder = Encoder(cfg)
    decoder = Decoder(cfg)
    refiner = Refiner(cfg)
    merger = Merger(cfg)

    if torch.cuda.is_available():
        encoder = torch.nn.DataParallel(encoder).cuda()
        decoder = torch.nn.DataParallel(decoder).cuda()
        refiner = torch.nn.DataParallel(refiner).cuda()
        merger = torch.nn.DataParallel(merger).cuda()

    print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
    checkpoint = torch.load(cfg.CONST.WEIGHTS)
    epoch_idx = checkpoint['epoch_idx']
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    if cfg.NETWORK.USE_REFINER:
        refiner.load_state_dict(checkpoint['refiner_state_dict'])
    if cfg.NETWORK.USE_MERGER:
        merger.load_state_dict(checkpoint['merger_state_dict'])

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()
    refiner.eval()
    merger.eval()

    with torch.no_grad():
        # Get data from data loader
        rendering_images = network_utils.var_or_cuda(inp)

        # Test the encoder, decoder, refiner and merger
        # [1, 1, 3, 224, 224]
        image_features = encoder(rendering_images)
        raw_features, generated_volume = decoder(image_features)

        if cfg.NETWORK.USE_MERGER:
            generated_volume = merger(raw_features, generated_volume)
        else:
            generated_volume = torch.mean(generated_volume, dim=1)

        if cfg.NETWORK.USE_REFINER:
            generated_volume = refiner(generated_volume)

        # [1, 32, 32, 32]

        # Volume Visualization
        gv = generated_volume.cpu().numpy()[0]
        # rendering_views = utils.binvox_visualization.get_volume_views(gv, 'test', epoch_idx)
    return gv


def get_args_from_command_line():
    parser = ArgumentParser(description='Parser of Runner of Pix2Vox')
    parser.add_argument('--gpu',
                        dest='gpu_id',
                        help='GPU device id to use [cuda0]',
                        default=cfg.CONST.DEVICE,
                        type=str)
    parser.add_argument('--rand', dest='randomize', help='Randomize (do not use a fixed seed)', action='store_true')
    parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
    parser.add_argument('--infer', dest='infer', help='Infer neural networks', action='store_true')
    parser.add_argument('--batch-size',
                        dest='batch_size',
                        help='name of the net',
                        default=cfg.CONST.BATCH_SIZE,
                        type=int)
    parser.add_argument('--epoch', dest='epoch', help='number of epoches', default=cfg.TRAIN.NUM_EPOCHES, type=int)
    parser.add_argument('--weights', dest='weights', help='Initialize network from the weights file', default=None)
    parser.add_argument('--out', dest='out_path', help='Set output path', default=cfg.DIR.OUT_PATH)
    parser.add_argument('--img', '--list', nargs='+', dest='img', help='Path to images', required=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args_from_command_line()

    if args.gpu_id is not None:
        cfg.CONST.DEVICE = args.gpu_id
    if not args.randomize:
        np.random.seed(cfg.CONST.RNG_SEED)
    if args.batch_size is not None:
        cfg.CONST.BATCH_SIZE = args.batch_size
    if args.epoch is not None:
        cfg.TRAIN.NUM_EPOCHES = args.epoch
    if args.out_path is not None:
        cfg.DIR.OUT_PATH = args.out_path
    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights
        if not args.test:
            cfg.TRAIN.RESUME_TRAIN = True

    # Set GPU to use
    if type(cfg.CONST.DEVICE) == str:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE

    inp = []
    for path in args.img:
        img = Image.open(path)
        img = np.array(img, dtype=np.float32)[None, :, :, :3] / 255
        inp.append(img)
    inp = np.concatenate(inp)
    volume = infer(cfg, inp)
    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(volume)

    print(verts.shape)
    print(faces.shape)

    initial_mesh = {
       'pos_idx': torch.tensor(faces.copy()), 
       'vtx_pos': (torch.tensor(verts.copy()) - 16) / 32, 
       'col_idx': torch.tensor(faces.copy()),
       'vtx_col': torch.ones(verts.shape)}

    fit_mesh.fit_mesh(initial_mesh, None)

    # visualize mesh
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
    # download weights from: https://github.com/hzxie/Pix2Vox/tree/Pix2Vox-F
    # put weights in submodules/Pix2Vox/pretrained
    # run: python estimate_shape.py --weights submodules/Pix2Vox/pretrained/Pix2Vox-A-ShapeNet.pth --img chair.png
    main()