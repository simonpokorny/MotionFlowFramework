import os.path
import pickle

import numpy as np

from datasets.base import BaseDataset


class WaymoDataset(BaseDataset):
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

        # Config parameters
        metadata_path = os.path.join(data_path, 'metadata')
        # It has information regarding the files and transformations

        self.data_path = data_path

        try:
            with open(metadata_path, 'rb') as metadata_file:
                self.metadata = pickle.load(metadata_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata not found, please create it by running preprocess.py. file : {metadata_path}")

        self._n_points = n_points

    def __len__(self):
        """
        For each dataset should be separetly written.
        Returns:
            length of the dataset
        """
        return len(self.metadata['look_up_table'])

    def _get_point_cloud_pair(self, index):
        """
        Read from disk the current and previous point cloud given an index
        """
        # In the lookup table entries with (current_frame, previous_frame) are stored
        current_frame = np.load(os.path.join(self.data_path, self.metadata['look_up_table'][index][1][0]))['frame']
        previous_frame = np.load(os.path.join(self.data_path, self.metadata['look_up_table'][index][0][0]))['frame']
        return previous_frame, current_frame

    def _get_pose_transform(self, index):
        """
        For each dataset should be separetly written. Returns transforamtion from t0 to t1
        Returns:
            t0_to_t1: in shape [4, 4]
        """

        current_frame_pose = self.metadata['look_up_table'][index][0][1]
        previous_frame_pose = self.metadata['look_up_table'][index][1][1]

        # G_T_C -> Global_TransformMatrix_Current
        G_T_C = np.reshape(np.array(current_frame_pose), [4, 4])

        # G_T_P -> Global_TransformMatrix_Previous
        G_T_P = np.reshape(np.array(previous_frame_pose), [4, 4])

        # Transformation matrix Previous (t0) to Current (t1)
        P_T_C = np.linalg.inv(G_T_P) @ G_T_C

        return P_T_C

    def _get_global_transform(self, index):
        """
        Optional. For each dataset should be separetly written. Returns transforamtion from t0 to
        global coordinates system.
        """
        previous_frame_pose = self.metadata['look_up_table'][index][1][1]

        # G_T_P -> Global_TransformMatrix_Previous
        G_T_P = np.reshape(np.array(previous_frame_pose), [4, 4])
        return G_T_P

    def _get_flow(self, index):
        """
        Optional. For each dataset should be separetly written. Returns gt flow in shape [N, channels].
        """
        previous_frame = np.load(os.path.join(self.data_path, self.metadata['look_up_table'][index][0][0]))['frame']
        flows = - (previous_frame[:, -4:-1] / 10)
        return flows