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


def get_pointwise_pillar_coords(batch_grid, n_pillars_x=640, n_pillars_y=640):
    """
    A method that takes a batch of grid indices in flatten mode and returns the corresponding 2D grid
    coordinates. The method calculates the x and y indices of the grid points using the number of
    pillars in the x and y dimensions, respectively, and then concatenates them along the second dimension.

    :param:
        - batch_grid: torch tensor of shape (BS, num_points)
        - n_pillars_x: size in x in bev
        - n_pillars_y: size in y in bev
    :return:
        - grid: torch tensor in shape (BS, num_points, 2)
    """
    assert batch_grid.ndim == 2

    grid = torch.cat(((batch_grid // n_pillars_x).unsqueeze(1),
                      (batch_grid % n_pillars_y).unsqueeze(1)), dim=1)
    return grid.transpose(-1, 1)

def create_bev_occupancy_grid(pointwise_pillar_coords, batch_mask, n_pillars_x=640, n_pillars_y=640):
    """
    A method that takes a batch of grid indices and masks and returns a tensor with a 1 in the location
    of each grid point and a 0 elsewhere. The method creates a tensor of zeros with the same shape as
    the voxel grid, and then sets the locations corresponding to the grid points in the batch to 1.
    """
    assert pointwise_pillar_coords.ndim == 3
    assert batch_mask.ndim == 2

    bs = pointwise_pillar_coords.shape[0]
    # pillar mask
    pillar_mask = torch.zeros((bs, 1, n_pillars_x, n_pillars_y), device=pointwise_pillar_coords.device)

    x = pointwise_pillar_coords[batch_mask][..., 0]
    y = pointwise_pillar_coords[batch_mask][..., 1]
    pillar_mask[:, :, x, y] = 1
    return pillar_mask