import os
import pickle

import numpy as np
from torch.utils.data import Dataset


class WaymoSeqDataset(Dataset):
    """
    Base class for all datasets.
    """

    def __init__(self, data_path,
                 point_cloud_transform=None,
                 drop_invalid_point_function=None,
                 n_points=None,
                 apply_pillarization=True):
        super().__init__()
        self._n_points = n_points
        self.data_path = data_path

        # Optional functions to apply to point clouds
        self._drop_invalid_point_function = drop_invalid_point_function
        self._point_cloud_transform = point_cloud_transform

        # This parameter is useful when visualizing, since we need to pass
        # the pillarized point cloud to the model for infer but we would
        # like to display the points without pillarizing them
        self._apply_pillarization = apply_pillarization

        # Config parameters
        metadata_path = os.path.join(data_path, 'metadata_seq')
        # It has information regarding the files and transformations

        self.data_path = data_path

        try:
            with open(metadata_path, 'rb') as metadata_file:
                self.metadata = pickle.load(metadata_file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Metadata not found, please create it by running preprocess.py. file : {metadata_path}")

    def __getitem__(self, index):
        """
        For appropriate function it is mandatory to overwrite  _get_point_cloud_pair and _get_pose_transform functions.

        Args:
            - index: Index for getitem

        Returns:
            - Tuple[t0_frame, t1_frame, t2_frame, t3frame], where t0_frame is Tuple[pcl, bev_projecticion]
              and t1_frame is Tuple[pcl, bev_projecticion].
            - flows: in shape [N, 4].
            - t0_to_t1: transformation matrix from t0 to t1.

        """

        # Mandatory
        t0_frame, t1_frame, t2_frame, t3_frame = self._get_point_cloud_pair(index)
        t3_to_t2, t3_to_t1, t3_to_t0 = self._get_pose_transform(index)

        # Drop invalid points according to the method supplied
        if self._drop_invalid_point_function is not None:

            t0_frame, _ = self._drop_invalid_point_function(t0_frame, None)
            t1_frame, _ = self._drop_invalid_point_function(t1_frame, None)
            t2_frame, _ = self._drop_invalid_point_function(t2_frame, None)
            t3_frame, _ = self._drop_invalid_point_function(t3_frame, None)


        # Subsample points based on n_points
        if self._n_points is not None:
            t0_frame = self._subsample_points(t0_frame)
            t1_frame = self._subsample_points(t1_frame)
            t2_frame = self._subsample_points(t2_frame)
            t3_frame = self._subsample_points(t3_frame)

        # Perform the pillarization of the point_cloud
        if self._point_cloud_transform is not None and self._apply_pillarization:
            t3_frame = self._point_cloud_transform(t3_frame)
            t2_frame = self._point_cloud_transform(t2_frame)
            t1_frame = self._point_cloud_transform(t1_frame)
            t0_frame = self._point_cloud_transform(t0_frame)
        else:
            # output must be a tuple
            t0_frame = (t0_frame, None)
            t1_frame = (t1_frame, None)
            t2_frame = (t2_frame, None)
            t3_frame = (t3_frame, None)

        return (t0_frame, t1_frame, t2_frame, t3_frame), (t3_to_t0, t3_to_t1, t3_to_t2)

    def __len__(self):
        """
        For each dataset should be separetly written.
        Returns:
            length of the dataset
        """
        return len(self.metadata['look_up_table'])
    def _get_point_cloud_pair(self, index):
        """
        For each dataset should be separetly written. Returns two consecutive point clouds.
        Args:
            index:

        Returns:
            t0_frame: pointcloud in shape [N, features]
            t1_frame: pointcloud in shape [N, features]
        """

        frame3 = np.load(os.path.join(self.data_path, self.metadata['look_up_table'][index][3][0]))['frame'][:, :5]
        frame2 = np.load(os.path.join(self.data_path, self.metadata['look_up_table'][index][2][0]))['frame'][:, :5]
        frame1 = np.load(os.path.join(self.data_path, self.metadata['look_up_table'][index][1][0]))['frame'][:, :5]
        frame0 = np.load(os.path.join(self.data_path, self.metadata['look_up_table'][index][0][0]))['frame'][:, :5]
        return frame0, frame1, frame2, frame3

    def _get_pose_transform(self, index):
        """
        For each dataset should be separetly written. Returns transforamtion from t0 to t1
        Returns:
            t0_to_t3: in shape [4, 4]
            t1_to_t3
            t2_to_t3
        """
        frame0 = self.metadata['look_up_table'][index][0][1]
        frame1 = self.metadata['look_up_table'][index][1][1]
        frame2 = self.metadata['look_up_table'][index][2][1]
        frame3 = self.metadata['look_up_table'][index][3][1]

        # G_T_C -> Global_TransformMatrix_Current
        G_T_0 = np.reshape(np.array(frame0), [4, 4])
        G_T_1 = np.reshape(np.array(frame1), [4, 4])
        G_T_2 = np.reshape(np.array(frame2), [4, 4])
        G_T_3 = np.reshape(np.array(frame3), [4, 4])

        # Transformation matrix Previous (t0) to Current (t1)
        C_T_P = (np.linalg.inv(G_T_3) @ G_T_2).astype(np.float32)
        C_T_PP = (np.linalg.inv(G_T_3) @ G_T_1).astype(np.float32)
        C_T_PPP = (np.linalg.inv(G_T_3) @ G_T_0).astype(np.float32)
        return C_T_P, C_T_PP, C_T_PPP

    def _get_global_transform(self, index):
        """
        Optional. For each dataset should be separetly written. Returns transforamtion from t0 to
        global coordinates system.
        """
        return None

    def _get_flow(self, index):
        """
        Optional. For each dataset should be separetly written. Returns gt flow in shape [N, channels].
        """
        return None

    def _subsample_points(self, frame):
        # current_frame.shape[0] == flows.shape[0]
        if frame.shape[0] > self._n_points:
            indexes_previous_frame = np.linspace(0, frame.shape[0] - 1, num=self._n_points).astype(int)
            frame = frame[indexes_previous_frame, :]
        return frame


if __name__ == "__main__":

    def cart2hom(pcl):
        assert pcl.ndim == 2 and pcl.shape[1] == 3, "PointCloud should be in shape [N, 3]"
        N, _ = pcl.shape
        return np.concatenate((pcl, np.ones((N, 1))), axis=1)

    def hom2cart(pcl):
        assert pcl.ndim == 2 and pcl.shape[1] == 4, "PointCloud should be in shape [N, 4]"
        return pcl[:, :3] / pcl[:, 3:4]


    ds = WaymoSeqDataset(data_path="../../data/waymoflow/train", n_points=8196, apply_pillarization=False)


    a = ds.__getitem__(0)
    b = None


    import open3d as o3d


    pcl0 = a[0][0][0][:, :3]
    pcl1 = a[0][1][0][:, :3]
    pcl2 = a[0][2][0][:, :3]
    pcl3 = a[0][3][0][:, :3]

    pcl0 = hom2cart((a[1][0] @ cart2hom(pcl0).T).T)
    pcl1 = hom2cart((a[1][1] @ cart2hom(pcl1).T).T)
    pcl2 = hom2cart((a[1][2] @ cart2hom(pcl2).T).T)


    # Assuming your tensor point cloud is named "tensor_point_cloud"
    o3d_point_cloud2 = o3d.geometry.PointCloud()
    o3d_point_cloud2.points = o3d.utility.Vector3dVector(pcl2)
    o3d_point_cloud2.paint_uniform_color([0.9, 0.1, 0.1])

    o3d_point_cloud3 = o3d.geometry.PointCloud()
    o3d_point_cloud3.points = o3d.utility.Vector3dVector(pcl3)
    o3d_point_cloud3.paint_uniform_color([0.1, 0.1, 0.7])

    o3d_point_cloud1 = o3d.geometry.PointCloud()
    o3d_point_cloud1.points = o3d.utility.Vector3dVector(pcl1)
    o3d_point_cloud1.paint_uniform_color([0.1, 0.1, 0.7])

    o3d_point_cloud0 = o3d.geometry.PointCloud()
    o3d_point_cloud0.points = o3d.utility.Vector3dVector(pcl0)
    o3d_point_cloud0.paint_uniform_color([0.1, 0.5, 0.1])



    o3d.visualization.draw_geometries([o3d_point_cloud0,o3d_point_cloud1,o3d_point_cloud3, o3d_point_cloud2])