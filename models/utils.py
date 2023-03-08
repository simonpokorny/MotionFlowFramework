import matplotlib.pyplot as plt
import torch
import numpy as np
from argparse import ArgumentTypeError


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def init_weights(m) -> None:
    """
    Apply the weight initialization to a single layer.
    Use this with your_module.apply(init_weights).
    The single layer is a module that has the weights parameter. This does not yield for all modules.
    :param m: the layer to apply the init to
    :return: None
    """
    if type(m) in [torch.nn.Linear, torch.nn.Conv2d]:
        # Note: There is also xavier_normal_ but the paper does not state which one they used.
        torch.nn.init.xavier_uniform_(m.weight)

def construct_batched_cuda_grid(pts, feature, x_min=-35, y_min=-35, grid_size=640):
    '''
    Assumes BS x N x CH (all frames same number of fake pts with zeros in the center)
    :param pts:
    :param feature:
    :param cfg:
    :return:
    '''
    BS = len(pts)
    bs_ind = torch.cat(
        [bs_idx * torch.ones(pts.shape[1], dtype=torch.long, device=pts.device) for bs_idx in range(BS)])

    feature_grid = - torch.ones(BS, grid_size, grid_size, device=pts.device).long()

    cell_size = torch.abs(2 * torch.tensor(x_min / grid_size))

    coor_shift = torch.tile(torch.tensor((x_min, y_min), dtype=torch.float, device=pts.device), dims=(BS, 1, 1))

    feature_ind = ((pts[:, :, :2] - coor_shift) / cell_size).long()

    feature_grid[
        bs_ind, feature_ind.flatten(0, 1)[:, 0], feature_ind.flatten(0, 1)[:, 1]] = feature.flatten().long()

    return feature_grid

if __name__ == "__main__":

    a = torch.rand((7,1,640,640))
