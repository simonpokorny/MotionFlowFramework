import torch
from pytorch3d.ops.knn import knn_points


def NN_loss(x, y, weights=None, reduction='mean'):
    '''
    Unique Nearest Neightboors?
    :param x:
    :param y:
    :param weights:
    :param reduction:
    :return:
    '''
    lengths1 = torch.tensor([x.shape[1]], dtype=torch.long, device=x.device)
    lengths2 = torch.tensor([y.shape[1]], dtype=torch.long, device=y.device)

    x_nn = knn_points(x, y, lengths1=lengths1, lengths2=lengths2, K=1, norm=1)


    cham_x = x_nn.dists[..., 0]  # (N, P1)

    nearest_to_y = x_nn[1]

    #corespond_y = torch.index_select(y,1, x_nn[1][0, :, 0])


    ### MASKING ####
    # checkujeme jestli nejsou souradnice mimo grid ..
    if weights is not None:
        min_fov_dist_cloud = torch.min(torch.cat([x[:, :, 0] + 35,
                             x[:, :, 1] + 35,
                             35 - x[:, :, 0],
                             35 - x[:, :, 1]]), 0)[0]

        weights = weights * (min_fov_dist_cloud > 0)
        weights = weights * (nearest_to_y < torch.square(min_fov_dist_cloud))
        nn_loss = cham_x * weights
    else:
        nn_loss = cham_x

    #loss = self.huber_loss(err_sqr=nearest_dist_sqr_a__b) * weights__b

    # breakpoint()
    # this way the NN loss is calculated only on one point of nn?
    # print(x, y)
    # print(cham_x)

    # breakpoint()
    # else:
    #     cham_x = x_nn.dists[:N_pts_x, 0]  # (N, P1)
        # cham_y = y_nn.dists[:N_pts_x, 0]  # (N, P2)
        #
        # nearest_to_y = x_nn[1][:,N_pts_x]

    # TODO rozmyslet, jestli potrebujeme two-way chamfer distance

    # nn_loss = x - y[nearest_to_y]
    # print(y[:, nearest_to_y, :].shape)
    #nn_loss = cham_x
    # nn_loss = (cham_x + cham_y) / 2

    if reduction == 'mean':
        nn_loss = nn_loss.mean()
    elif reduction == 'sum':
        nn_loss = nn_loss.sum()
    elif reduction == 'none':
        nn_loss = nn_loss
    else:
        raise NotImplementedError

    # breakpoint()
    return nn_loss


def rigid_cycle_loss(p_i, fw_trans, bw_trans, reduction='mean'):
    """
    Computes the rigid cycle loss between two sets of 3D points p_i, after applying forward and backward rigid
    transformations fw_trans and bw_trans respectively.

    Args:
    - p_i (torch.Tensor): tensor of shape (batch_size, num_points, 3) representing the 3D points to transform.
    - fw_trans (torch.Tensor): tensor of shape (batch_size, 4, 4) representing the forward rigid transformation matrix.
    - bw_trans (torch.Tensor): tensor of shape (batch_size, 4, 4) representing the backward rigid transformation matrix.
    - reduction (str): specifies the reduction to apply to the loss. Possible values are 'none' (default), 'mean' and 'sum'.

    Returns:
    - loss (torch.Tensor): tensor of shape (batch_size,) representing the rigid cycle loss between the transformed points and the original points.
    """

    trans_p_i = torch.cat((p_i, torch.ones((len(p_i), p_i.shape[1], 1), device=p_i.device)), dim=2)
    bw_fw_trans = bw_trans @ fw_trans - torch.eye(4, device=fw_trans.device)
    loss = torch.matmul(bw_fw_trans, trans_p_i.permute(0, 2, 1)).norm(dim=1)

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    else:
        raise NotImplementedError
    return loss

def static_point_loss(pcl, T, static_flow, staticness, reduction="mean"):
    aggr_flow = torch.cat((pcl, torch.ones((*pcl.shape[:-1], 1), device=pcl.device)), dim=2)
    aggr_flow = (T.float() @ aggr_flow.transpose(1, 2)).transpose(1, 2)[:, :, :3]
    aggr_flow = aggr_flow - pcl
    loss = weighted_mse_loss(input=static_flow, target=aggr_flow,weight=staticness)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    else:
        raise NotImplementedError
    return loss


def smoothness_loss(p_i, est_flow, K, reduction='mean'):
    l = torch.tensor([p_i.shape[1]], dtype=torch.long, device=p_i.device)

    nn = knn_points(p_i, p_i, l, l, K=K + 1, norm=1)
    nn_indices = nn.idx[..., 1:]

    flow_nn = est_flow[torch.arange(est_flow.shape[0]), nn_indices, :]
    #loss = torch.sum(torch.square(flow_nn - est_flow.unsqueeze(-2)), -1).mean(dim=-1)
    loss = (flow_nn - est_flow.unsqueeze(-2)).square().sum(2).sum(2) / K

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    else:
        raise NotImplementedError
    return loss


def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2)


if __name__ == "__main__":
    pass
