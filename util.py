import torch


def projection(x=0.1, n=1.0., f=50.0):
    return torch.tensor([[n/x,    0.,            0.,              0],
                     [  0., n/-x,            0.,              0],
                     [  0.,    0., -(f+n)/(f-n), -(2*f*n)/(f-n)],
                     [  0.,    0.,           -1.,              0.]])

def translate(x=0., y=0., z=0.):
    return torch.tensor([[1., 0., 0., x],
                     [0., 1., 0., y],
                     [0., 0., 1, z],
                     [0., 0., 0., 1.]])