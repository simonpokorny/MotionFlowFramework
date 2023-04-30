import numpy as np
import torch.utils.data as data
import open3d as o3d

from .utils import show_flow, show_pcl


class Visualizer(data.DataLoader):
    def __init__(self, dataloader, visualize, batch_size=1, shuffle=False, dims=3):
        super().__init__(dataloader.dataset, batch_size=batch_size, shuffle=shuffle)
        assert dims in [2, 3], "Visualizer can only visualize in 2d or 3d space"

        types = ["flow3d", "sync3d", "sync2d", "flow2d"]
        if visualize == "flow3d":
            self.visualize = self.visualize3Dflow
        elif visualize == "flow2d":
            raise NotImplementedError()
        elif visualize == "sync3d":
            self.visualize = self.visualize3Dsync
        elif visualize == "sync2d":
            raise NotImplementedError()
        else:
            raise ValueError(f"visualize must be in {types}")


    def __iter__(self):
        # Get the original data loader iterator
        data_iter = super(Visualizer, self).__iter__()

        # iterate through the batches and plot them
        for batch_idx, batch_data in enumerate(data_iter):
            # Plot the batch data
            self.visualize(batch_data)
            yield batch_data


    def visualize3Dflow(self, batch):
        """
        Visualizes a pair of point clouds with flow between them.

        Args:
            index (int): The index of the frame pair to visualize.

        Returns:
            None
        """
        x, flow, t0_to_t1 = batch

        t0_frame = x[0][0].detach().cpu().numpy()[0]
        t1_frame = x[1][0].detach().cpu().numpy()[0]

        # Get only xyz coordinates without any features (unpillarization)
        t0_frame = t0_frame[:, :3] + t0_frame[:, 3:6]
        t1_frame = t1_frame[:, :3] + t1_frame[:, 3:6]

        flow = flow.detach().cpu().numpy()[0]
        assert (flow.shape == t0_frame.shape), "Flows should be equal to frame t0"

        t0_to_t1 = t0_to_t1.detach().cpu().numpy()[0]
        assert t0_to_t1.shape == (4, 4), "Matrix in custom dataset must be in shape (4, 4)"

        show_flow(t0_frame, t1_frame, flow)

    def visualize3Dsync(self, batch):
        """
        Visualizes a pair of point clouds with flow between them.

        Args:
            index (int): The index of the frame pair to visualize.

        Returns:
            None
        """
        x, flow, t0_to_t1 = batch

        t0_frame = x[0][0].detach().cpu().numpy()[0]
        t1_frame = x[1][0].detach().cpu().numpy()[0]

        t0_to_t1 = t0_to_t1.detach().cpu().numpy()[0]
        assert t0_to_t1.shape == (4, 4), "Matrix in custom dataset must be in shape (4, 4)"

        # Get only xyz coordinates without any features (unpillarization)
        t0_frame = t0_frame[:, :3] + t0_frame[:, 3:6]
        t1_frame = t1_frame[:, :3] + t1_frame[:, 3:6]

        t0_frame = self.cart2hom(t0_frame).T
        t0_frame = (t0_to_t1 @ t0_frame).T
        t0_frame = self.hom2cart(t0_frame)

        show_pcl(t0_frame, t1_frame)

    @staticmethod
    def cart2hom(pcl):
        assert pcl.ndim == 2 and pcl.shape[1] == 3, "PointCloud should be in shape [N, 3]"
        N, _ = pcl.shape
        return np.concatenate((pcl, np.ones((N, 1))), axis=1)

    @staticmethod
    def hom2cart(pcl):
        assert pcl.ndim == 2 and pcl.shape[1] == 4, "PointCloud should be in shape [N, 4]"
        return pcl[:, :3] / pcl[:, 3:4]
