import torch
from pytorch3d.ops.knn import knn_points

def NN_loss(x, y, x_lengths=None, y_lengths=None, reduction='mean'):
    '''
    Unique Nearest Neightboors?
    :param x:
    :param y:
    :param x_lengths:
    :param y_lengths:
    :param reduction:
    :return:
    '''
    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1, norm=1)
    # y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1, norm=1)


    cham_x = x_nn.dists[..., 0]  # (N, P1)
    # cham_y = y_nn.dists[..., 0]  # (N, P2)

    nearest_to_y = x_nn[1]
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
    nn_loss = cham_x
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
    return nn_loss, nearest_to_y

def rigid_cycle_loss(p_i, fw_trans, bw_trans, reduction='none'):

    trans_p_i = torch.cat((p_i, torch.ones((len(p_i), p_i.shape[1], 1), device=p_i.device)), dim=2)
    bw_fw_trans = bw_trans @ fw_trans - torch.eye(4, device=fw_trans.device)
    # todo check this in visualization, if the points are transformed as in numpy
    res_trans = torch.matmul(bw_fw_trans, trans_p_i.permute(0, 2, 1)).norm(dim=1)

    rigid_loss = res_trans.mean()

    return rigid_loss

if __name__ == "__main__":
    x = torch.rand(3, 10, 3, requires_grad=True)
    y = torch.rand(3, 15, 3, requires_grad=True)


    x_lengths = torch.ones(len(x), dtype=torch.long) * x.shape[1]
    y_lengths = torch.ones(len(y), dtype=torch.long) * x.shape[1]

    nn_loss, nn_x_to_y = NN_loss(x, y, x_lengths, y_lengths)

    nn_loss.backward()
