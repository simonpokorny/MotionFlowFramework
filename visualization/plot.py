import math
import os

import matplotlib.pyplot as plt
import numpy as np
#import open3d as o3d
import torch
from matplotlib.patches import Circle


'''
def visualise_tensor(tensor_batch, title="Plot of tensor"):
    """
    Visualizes a PyTorch tensor in the shape [batch size, channels, height, width] as a grid of images using Matplotlib.

    Parameters:
        tensor_batch (torch.Tensor): The tensor to visualize, with shape [batch size, channels, height, width].
        title (str): The title of the plot (default: "Plot of tensor").

    Example:
        # Create a random tensor with 2 images, 3 channels, and 64x64 resolution
        tensor = torch.randn(2, 3, 64, 64)

        # Visualize the tensor as a grid of images
        visualise_tensor(tensor)
    """
    assert tensor_batch.ndim == 4
    # Get number of channels
    batch_size, channels, _, _ = tensor_batch.shape
    nrows = ncols = int(np.ceil(np.sqrt(batch_size)))

    # Convert the tensor to a numpy array and transpose the dimensions to [batch size, height, width, channels]
    tensor_np = tensor_batch.numpy().transpose(0, 2, 3, 1)

    # Create a figure with subplots for each image in the batch
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    # Loop over the images in the batch and plot them in the corresponding subplot
    for i in range(nrows * ncols):
        if i < batch_size:
            fig.axes[i].imshow(tensor_np[i])
        else:
            fig.axes[i].set_axis_off()

    # Show the plot
    plt.title(legend)
    plt.show()


def plot_pillars_heatmap(indices, x_max, x_min, y_max, y_min, grid_cell_size):
    fig = plt.figure(figsize=(15, 15))
    x, y = indices[:, 0], indices[:, 1]
    n_pillars_x = math.floor((x_max - x_min) / grid_cell_size)

    plt.hist2d(x, y, bins=n_pillars_x, cmap='YlOrBr', density=True)

    cb = plt.colorbar()
    cb.set_label('Number of points in pillar')

    plt.title('Heatmap of pillars (bev projection)')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.show()


def plot_pillars(indices, x_max, x_min, y_max, y_min, grid_cell_size):
    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection="3d")

    n_pillars_x = math.floor((x_max - x_min) / grid_cell_size)
    n_pillars_y = math.floor((y_max - y_min) / grid_cell_size)
    pillar_matrix = np.zeros(shape=(n_pillars_x, n_pillars_y, 1))

    for x, y in indices:
        pillar_matrix[x, y] += 1

    x_pos, y_pos, z_pos = [], [], []
    x_size, y_size, z_size = [], [], []

    for i in range(pillar_matrix.shape[0]):
        for j in range(pillar_matrix.shape[1]):
            x_pos.append(i * grid_cell_size)
            y_pos.append(j * grid_cell_size)
            z_pos.append(0)

            x_size.append(grid_cell_size)
            y_size.append(grid_cell_size)
            z_size.append(int(pillar_matrix[i, j]))

    ax.bar3d(x_pos, y_pos, z_pos, x_size, y_size, z_size)
    plt.title("3D projection of pillar map")
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.show()


def plot_tensor(tensor, tittle="None"):
    plt.imshow(tensor.numpy())
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title(tittle)
    plt.show()


def plot_2d_point_cloud(pc, title="2D projection of pointcloud"):
    fig, ax = plt.subplots(figsize=(15, 15))

    x, y = [], []
    for p in pc:
        x.append(p[0])
        y.append(p[1])
    ax.scatter(x, y, color="green")
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title(title)
    plt.show()


def visualize_point_cloud(points):
   """ Input must be a point cloud of shape (n_points, 3) """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([point_cloud])


def visualize_flows(vis, points, flows):
    """
    Visualize a 3D point cloud where is point is flow-color-coded
    :param vis: visualizer created with open3D, for example:

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)

    :param points: (n_points, 3)
    :param flows: (n_points, 3)
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(flows)
    # vis.destroy_window()
'''

def plot_pcl(*tensors, name, colors):
    if type(colors) is not list:
        colors = list(colors)
    fig, ax = plt.subplots(figsize=(30, 30))
    for tensor, color in zip(tensors, colors):
        assert tensor.ndim == 3
        pc = tensor[0].detach().cpu()
        x, y = [], []
        for p in pc:
            x.append(p[0])
            y.append(p[1])
        plt.scatter(y, x, color=color, s=0.05)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.ylabel('x axis')
    plt.xlabel('y axis')
    plt.title(name)


def save_pcl(tensor, path, name, color, show=False):
    plot_pcl(tensor, name=name, colors=color)
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(path, name))


def save_pcl_channel_color(tensor, path, name, index_for_color, show=False):
    assert tensor.ndim == 3
    fig, ax = plt.subplots(figsize=(30, 30))
    pc = tensor[0].detach().cpu()
    x, y, c = [], [], []
    for p in pc:
        x.append(p[0])
        y.append(p[1])
        c.append(p[index_for_color])
    plt.scatter(y, x, color=c, s=0.05)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.axis('equal')
    plt.ylabel('x axis')
    plt.xlabel('y axis')
    plt.title(name)
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(path, name))


def save_trans_pcl(P_T_C, P, C, path, name, show=False):
    assert P.ndim == 3
    assert C.ndim == 3

    P_T_C = P_T_C.detach().cpu()
    P = P.detach().cpu()
    C = C.detach().cpu()

    P = torch.concat((P[:, :, :3], torch.ones_like(P)[:, :, :1]), dim=2).permute(0, 2, 1)
    P_in_C = torch.einsum('bij,bjk->bik', P_T_C.double(), P.double()).permute(0, 2, 1).float()

    plot_pcl(P_in_C, C, name=name, colors=["r", "b"])
    plt.axis('equal')
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(path, name), dpi=200)


def save_tensor(tensor, path, name, show=False):
    assert tensor.ndim == 4
    # Get number of channels
    batch_size, channels, _, _ = tensor.shape
    assert batch_size == 1

    # Convert the tensor to a numpy array and transpose the dimensions to [batch size, height, width, channels]
    tensor = tensor.permute(0, 2, 3, 1).detach().cpu().numpy()[0]

    # Create a figure with subplots for each image in the batch
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))

    plt.imshow(tensor)
    plt.title(name)
    plt.colorbar()
    axes.axis('equal')
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(path, name))


def save_tensor_per_channel(tensor, path, name, labels=None, show=False):
    assert tensor.ndim == 4
    # Get number of channels
    batch_size, channels, _, _ = tensor.shape
    #assert channels == 3
    assert batch_size == 1

    # Convert the tensor to a numpy array and transpose the dimensions to [batch size, height, width, channels]
    tensor = tensor.detach().cpu().numpy()

    # Create a figure with subplots for each image in the batch
    fig, axes = plt.subplots(nrows=channels, ncols=1, figsize=(15, 15))

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # Loop over the images in the batch and plot them in the corresponding subplot
    for i, logits in enumerate(tensor):
        for j, logit in enumerate(logits):
            fig.axes[j].set_title(labels[j]) if labels is not None else None
            im = fig.axes[j].imshow(logit)
            divider = make_axes_locatable(fig.axes[j])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
    axes.axis('equal')
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(path, name))



def save_pcl_class(tensor, classes, path, name, colors, labels, show=False):
    fig, ax = plt.subplots(figsize=(30, 30))
    assert tensor.ndim == 3
    pc = tensor[0].detach().cpu()
    cls = classes[0]
    x, y = [], []
    #labels = ["static", "dynamic", "ground"]
    #colors = ["r", "g", "b"]
    for i in range(len(colors)):
        act_pc = pc[cls == i]
        plt.scatter(act_pc[:,1], act_pc[:,0], c=colors[i], s=0.5, label=labels[i])

    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.legend()
    plt.title(name)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.axis('equal')
    if show:
        plt.show()
    else:
        plt.savefig(os.path.join(path, name))

def save_pcl_flow(pcl0, pcl1, flow, odom, path, name, show=False, num_points=1000, size=4):
    fig, ax = plt.subplots()
    ax.invert_yaxis()
    fig.set_size_inches(60, 60)

    flow = flow[0].detach().cpu()
    odom = odom[0,:2,3].detach().cpu()
    pcl0 = (pcl0[0, :, :3] + pcl0[0, :, 3:6]).detach().cpu()
    pcl1 = (pcl1[0, :, :3] + pcl1[0, :, 3:6]).detach().cpu()

    indices = np.arange(min(pcl0.shape[0], pcl1.shape[0]))
    indices = np.random.choice(indices, min(pcl0.shape[0], pcl1.shape[0], int(num_points)))

    pcl0 = pcl0[indices]
    pcl1 = pcl1[indices]
    flow = flow[indices]

    plt.scatter(pcl0[:, 1], pcl0[:, 0], s=2, c="r")
    plt.scatter(pcl1[:, 1], pcl1[:, 0], s=2, c="b")
    for x, y, dx, dy in zip(pcl0[:, 1], pcl0[:, 0], flow[:, 1], flow[:, 0]):
        plt.arrow(x=x, y=y, dx=dx, dy=dy)

    circle = Circle((0, -35), radius=size, facecolor="white", edgecolor='black')
    ax.add_patch(circle)
    ax.axis('equal')

    _odom = (odom.numpy() / np.linalg.norm(odom.numpy())) * size
    plt.arrow(x=0, y=-35, dx=_odom[1], dy=_odom[0], linewidth=4, head_width=0.1)
    plt.arrow(x=0, y=-35, dx= odom[1], dy=odom[0], linewidth=4, color="red")
    plt.title(name)
    if show:
        plt.show(dpi=200)
    else:
        plt.savefig(os.path.join(path, name), dpi=200)

def show_pcl(pcl):
    import open3d as o3d
    pcl = pcl[:, :, :3] + pcl[:, :, 3:6]
    # Assuming your tensor point cloud is named "tensor_point_cloud"
    o3d_point_cloud = o3d.geometry.PointCloud()
    o3d_point_cloud.points = o3d.utility.Vector3dVector(pcl[0].numpy())
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(o3d_point_cloud)
    visualizer.run()

def show_flow(pcl, flow, pcl2):
    import open3d as o3d
    pcl = pcl[:, :, :3] + pcl[:, :, 3:6]
    pcl2 = pcl2[:, :, :3] + pcl2[:, :, 3:6]

    flow = torch.cat([pcl, pcl + flow], dim=1)
    indices = torch.cat([torch.reshape(torch.arange(pcl.shape[1]), (1,-1)),
                         torch.reshape(torch.arange(pcl.shape[1], pcl.shape[1]*2), (1,-1))], dim=0)

    # Assuming your tensor point cloud is named "tensor_point_cloud"
    o3d_point_cloud = o3d.geometry.PointCloud()
    o3d_point_cloud.points = o3d.utility.Vector3dVector(pcl[0].numpy())
    o3d_point_cloud.paint_uniform_color([0.9, 0.1, 0.1])

    o3d_point_cloud2 = o3d.geometry.PointCloud()
    o3d_point_cloud2.points = o3d.utility.Vector3dVector(pcl2[0].numpy())
    o3d_point_cloud2.paint_uniform_color([0.1, 0.1, 0.7])


    o3d_flow = o3d.geometry.LineSet()
    o3d_flow.points = o3d.utility.Vector3dVector(flow[0].numpy())
    o3d_flow.lines = o3d.utility.Vector2iVector(indices.T.numpy())

    o3d.visualization.draw_geometries([o3d_point_cloud, o3d_flow, o3d_point_cloud2])

    #visualizer = o3d.visualization.Visualizer()
    #visualizer.create_window()
    #visualizer.add_geometry([o3d_point_cloud, o3d_flow, o3d_point_cloud2])
    #visualizer.run()

if __name__ == "__main__":

    # COMPUTE NUM POINTS FOR MOVING DYNAMIC THRESHOLDS

    import numpy as np
    from tqdm import tqdm
    from datasets.waymoflow.WaymoDataModule import WaymoDataModule

    grid_cell_size = 0.109375
    dataset_path = "../data/waymoflow_subset/"

    data_module = WaymoDataModule(
        dataset_directory=dataset_path,
        grid_cell_size=grid_cell_size,
        x_min=-35,
        x_max=35,
        y_min=-35,
        y_max=35,
        z_min=0.25,
        z_max=10,
        batch_size=1,
        has_test=False,
        num_workers=0,
        n_pillars_x=640,
        n_points=20000,
        apply_pillarization=True)

    data_module.setup()
    train_dl = data_module.train_dataloader()


    for x, flow, T_gt in tqdm(train_dl):
        # Create pcl from features vector

        pcl1 = x[0][0]
        pcl2 = x[1][0]
        flow = flow[ :,:, :3]

        show_flow(pcl2, flow, pcl1)




