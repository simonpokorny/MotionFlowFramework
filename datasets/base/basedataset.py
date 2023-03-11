import numpy as np
from torch.utils.data import Dataset


class BaseDataset(Dataset):
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

    def __getitem__(self, index):
        """
        For appropriate function it is mandatory to overwrite  _get_point_cloud_pair and _get_pose_transform functions.

        Args:
            - index: Index for getitem

        Returns:
            - Tuple[t0_frame, t1_frame], where t0_frame is Tuple[pcl, bev_projecticion]
              and t1_frame is Tuple[pcl, bev_projecticion].
            - flows: in shape [N, 4].
            - t0_to_t1: transformation matrix from t0 to t1.

        """

        # Mandatory
        t1_frame, t0_frame = self._get_point_cloud_pair(index)
        t0_to_t1 = self._get_pose_transform(index)

        # (optional) ground truth flows
        flows = self._get_flow(index)

        # Drop invalid points according to the method supplied
        if self._drop_invalid_point_function is not None:
            print(type(t1_frame), type(flows))
            t1_frame, flows = self._drop_invalid_point_function(t1_frame, None)
            t0_frame, _ = self._drop_invalid_point_function(t0_frame, None)

        # Perform the pillarization of the point_cloud
        if self._point_cloud_transform is not None and self._apply_pillarization:
            t1_frame = self._point_cloud_transform(t1_frame)
            t0_frame = self._point_cloud_transform(t0_frame)
        else:
            # output must be a tuple
            t0_frame = (t0_frame, None)
            t1_frame = (t1_frame, None)

        return (t0_frame, t1_frame), flows, t0_to_t1

    def __len__(self):
        """
        For each dataset should be separetly written.
        Returns:
            length of the dataset
        """
        raise NotImplementedError()

    def _get_point_cloud_pair(self, index):
        """
        For each dataset should be separetly written. Returns two consecutive point clouds.
        Args:
            index:

        Returns:
            t1_frame: pointcloud in shape [N, features]
            t0_frame: pointcloud in shape [N, features]
        """

        raise NotImplementedError()

    def _get_pose_transform(self, index):
        """
        For each dataset should be separetly written. Returns transforamtion from t0 to t1
        Returns:
            t0_to_t1: in shape [4, 4]
        """
        raise NotImplementedError()

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

    def _subsample_points(self, frame_t1, frame_t0, flows):
        # current_frame.shape[0] == flows.shape[0]
        if frame_t1.shape[0] > self._n_points:
            indexes_current_frame = np.linspace(0, frame_t1.shape[0] - 1, num=self._n_points).astype(int)
            frame_t1 = frame_t1[indexes_current_frame, :]
            flows = flows[indexes_current_frame, :]
        if frame_t0.shape[0] > self._n_points:
            indexes_previous_frame = np.linspace(0, frame_t0.shape[0] - 1, num=self._n_points).astype(int)
            frame_t0 = frame_t0[indexes_previous_frame, :]
        return frame_t1, frame_t0, flows
