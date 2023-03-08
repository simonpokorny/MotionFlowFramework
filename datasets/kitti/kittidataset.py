import glob
import os.path

import numpy as np

from datasets.base import BaseDataset


class KittiRawDataset(BaseDataset):
    def __init__(self, data_path,
                 drop_invalid_point_function=None,
                 point_cloud_transform=None,
                 n_points=None,
                 apply_pillarization=True):
        super().__init__(drop_invalid_point_function,
                         point_cloud_transform,
                         n_points,
                         apply_pillarization)

        self.files = glob.glob(os.path.join(data_path, "*/*/pairs/*.npz"))
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
        Args:
            index:

        Returns:
            t1_frame: pointcloud in shape [N, features]
            t0_frame: pointcloud in shape [N, features]
        """
        self.frame = np.load(self.files[0])
        return self.frame['pcl_t1'], self.frame['pcl_t0']

    def _get_pose_transform(self, index):
        """
        For each dataset should be separetly written. Returns transforamtion from t0 to t1
        Returns:
            t0_to_t1: in shape [4, 4]
        """
        return self.frame['odom_t0_t1']

    def _get_global_transform(self, index):
        """
        Optional. For each dataset should be separetly written. Returns transforamtion from t0 to
        global coordinates system.
        """
        return self.frame['global_pose']

    def _get_flow(self, index):
        """
        Optional. For each dataset should be separetly written. Returns gt flow in shape [1, N, channels].
        """
        return None
