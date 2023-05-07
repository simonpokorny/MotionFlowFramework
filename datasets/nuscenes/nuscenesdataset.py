import glob
import os.path

import numpy as np

from datasets.base import BaseDataset


class NuScenesDataset(BaseDataset):
    def __init__(self, data_path,
                 point_cloud_transform=None,
                 drop_invalid_point_function=None,
                 n_points=None,
                 apply_pillarization=True,
                 point_features=3):
        super().__init__(data_path=data_path,
                         point_cloud_transform=point_cloud_transform,
                         drop_invalid_point_function=drop_invalid_point_function,
                         n_points=n_points,
                         apply_pillarization=apply_pillarization,
                         point_features=point_features)

        self.files = glob.glob(os.path.join(data_path, "*.npz"))
        self.frame = None

    def __len__(self):
        """
        For each dataset should be separetly written.
        Returns:
            length of the dataset
        """
        return len(self.files)

    def _get_point_cloud_pair(self, index):
        """
        For each dataset should be separetly written. Returns two consecutive point clouds.
        Pcl are swopped because odometry is also swapped.
        Args:
            index:

        Returns:
            t0_frame: pointcloud in shape [N, features]
            t1_frame: pointcloud in shape [N, features]
        """
        self.frame = np.load(self.files[index])
        return self.frame['pcl_t0'], self.frame['pcl_t1']

    def _get_pose_transform(self, index):
        """
        For each dataset should be separetly written. Returns transforamtion from t0 to t1
        Returns:
            t0_to_t1: in shape [4, 4]
        """
        self.frame = np.load(self.files[index])
        return np.linalg.inv(self.frame['odom_t0_t1'])

    def _get_global_transform(self, index):
        """
        Optional. For each dataset should be separetly written. Returns transforamtion from t0 to
        global coordinates system.
        """
        return self.frame['global_pose']

    def _get_flow(self, index):
        """
        Optional. For each dataset should be separetly written. Returns gt flow in shape [N, channels].
        """
        return self.frame["flow_t0_t1"]


