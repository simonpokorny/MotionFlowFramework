import glob
import os.path

import numpy as np

from datasets.base import BaseDataset


class KittiSceneFlowDataset(BaseDataset):
    def __init__(self, data_path,
                 point_cloud_transform=None,
                 drop_invalid_point_function=None,
                 n_points=None,
                 apply_pillarization=True):
        super().__init__(data_path=data_path,
                         point_cloud_transform=point_cloud_transform,
                         drop_invalid_point_function=drop_invalid_point_function,
                         n_points=n_points,
                         apply_pillarization=apply_pillarization)

        self.files = glob.glob((os.path.join(data_path, "*.npz")))
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
            t0_frame: pointcloud in shape [N, features]
            t1_frame: pointcloud in shape [N, features]
        """
        self.frame = np.load(self.files[index])

        x, z, y = self.frame['pc1'].T
        pc1 = np.stack((x, y, z)).T

        x, z, y = self.frame['pc2'].T
        pc2 = np.stack((x, y, z)).T
        return pc1, pc2

    def _get_pose_transform(self, index):
        """
        For each dataset should be separetly written. Returns transforamtion from t0 to t1
        Returns:
            t0_to_t1: in shape [4, 4]
        """
        return None

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
        self.frame = np.load(self.files[index])

        x, z, y = self.frame['flow'].T
        flow = np.stack((x, y, z)).T
        return flow
