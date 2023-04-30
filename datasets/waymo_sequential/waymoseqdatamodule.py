"""
Source: https://github.com/Jabb0/FastFlow3D
"""
import sys

sys.path.append("../")
sys.path.append("../../")

from pathlib import Path
from typing import Optional, Union, List, Dict

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets.waymo_sequential.waymoseqdataset import WaymoSeqDataset
from datasets.waymoflow.util import ApplyPillarization, drop_points_function, _pad_batch, _pad_targets, default_collate

def custom_collate_batch(batch):
    """
    This version of the collate function create the batch necessary for the input to the network.

    Take the list of entries and batch them together.
        This means a batch of the previous images and a batch of the current images and a batch of flows.
    Because point clouds have different number of points the batching needs the points clouds with less points
        being zero padded.
    Note that this requires to filter out the zero padded points later on.

    :param batch: batch_size long list of ((prev, cur), flows) pointcloud tuples with flows.
        prev and cur are tuples of (point_cloud, grid_indices, mask)
         point clouds are (N_points, features) with different N_points each
    :return: ((batch_prev, batch_cur), batch_flows)
    """
    # Build numpy array with data

    # Only convert the points clouds from numpy arrays to tensors
    # entry[0, 0] is the previous (point_cloud, grid_index) entry
    batch_0 = [
        entry[0][0] for entry in batch
    ]
    batch_0 = _pad_batch(batch_0)

    batch_1 = [
        entry[0][1] for entry in batch
    ]
    batch_1 = _pad_batch(batch_1)

    batch_2 = [
        entry[0][2] for entry in batch
    ]
    batch_2 = _pad_batch(batch_2)

    batch_3 = [
        entry[0][3] for entry in batch
    ]
    batch_3 = _pad_batch(batch_3)

    # For the targets we can only transform each entry to a tensor and not stack them

    batch_transform = [
        entry[1] for entry in batch
    ]


    # Call the default collate to stack everything
    batch_0 = default_collate(batch_0)
    batch_1 = default_collate(batch_1)
    batch_2 = default_collate(batch_2)
    batch_3 = default_collate(batch_3)

    batch_transform = default_collate(batch_transform)

    # Return a tensor that consists of
    # the data batches consist of batches of tensors
    #   1. (batch_size, max_n_points, features) the point cloud batch
    #   2. (batch_size, max_n_points) the 1D grid_indices encoding to map to
    #   3. (batch_size, max_n_points) the 0-1 encoding if the element is padded
    #   4. (batch_size, 4, 4) transformation matrix from frame to global coords
    # Batch previous for the previous frame
    # Batch current for the current frame

    # The targets consist of
    #   (batch_size, max_n_points, target_features). should by 4D x,y,z flow and class id

    return (batch_0, batch_1, batch_2, batch_3), batch_transform

class WaymoSeqDataModule(pl.LightningDataModule):
    """
    Data module to prepare and load the waymo dataset.
    Using a data module streamlines the data loading and preprocessing process.
    """
    def __init__(self, dataset_directory,
                 # These parameters are specific to the dataset
                 grid_cell_size, x_min, x_max, y_min, y_max, z_min, z_max, n_pillars_x, n_pillars_y=640,
                 batch_size: int = 32,
                 has_test=False,
                 num_workers=1,
                 n_points=None,
                 apply_pillarization=True,
                 shuffle_train=True,
                 point_features=6):
        super(WaymoSeqDataModule, self).__init__()
        self._dataset_directory = Path(dataset_directory)
        self._batch_size = batch_size
        self._train_ = None
        self._val_ = None
        self._test_ = None
        self._shuffle_train = shuffle_train
        self.apply_pillarization = apply_pillarization

        self._pillarization_transform = ApplyPillarization(grid_cell_size=grid_cell_size, x_min=x_min,
                                                           y_min=y_min, z_min=z_min, z_max=z_max,
                                                           n_pillars_x=n_pillars_x)

        # This returns a function that removes points that should not be included in the pillarization.
        # It also removes the labels if given.
        self._drop_points_function = drop_points_function(x_min=x_min,
                                                          x_max=x_max, y_min=y_min, y_max=y_max,
                                                          z_min=z_min, z_max=z_max)
        self._has_test = has_test
        self._num_workers = num_workers

        self._collate_fn = custom_collate_batch
        self._n_points = n_points

    def prepare_data(self) -> None:
        """
        Preprocessing of the data only called on 1 GPU.
        Download and process the datasets here. E.g., tokenization.
        Everything that is not random and only necessary once.
        This is used to download the dataset to a local storage for example.
            Later the dataset is then loaded by every worker in the setup() method.
        :return: None
        """
        # No need to download stuff
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup of the datasets. Called on every GPU in distributed training.
        Do splits and build model internals here.
        :param stage: either 'fit', 'validate', 'test' or 'predict'
        :return: None
        """
        self._train_ = WaymoSeqDataset(self._dataset_directory.joinpath("train"),
                                    point_cloud_transform=self._pillarization_transform,
                                    drop_invalid_point_function=self._drop_points_function,
                                    n_points=self._n_points,
                                    apply_pillarization=self.apply_pillarization)

        if self._has_test:
            self._test_ = WaymoSeqDataset(self._dataset_directory.joinpath("test"),
                                       point_cloud_transform=self._pillarization_transform,
                                       drop_invalid_point_function=self._drop_points_function,
                                       n_points=self._n_points,
                                       apply_pillarization=self.apply_pillarization
                                       )

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """
        Return a data loader for training
        :return: the dataloader to use
        """
        return DataLoader(self._train_, self._batch_size, num_workers=self._num_workers,
                          shuffle=self._shuffle_train,
                          collate_fn=self._collate_fn)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """
        Return a data loader for testing
        :return: the dataloader to use
        """
        if not self._has_test:
            raise RuntimeError("No test dataset specified. Maybe set has_test=True in DataModule init.")
        return DataLoader(self._test_, self._batch_size, shuffle=False, num_workers=self._num_workers,
                          collate_fn=self._collate_fn)


if __name__ == "__main__":
    # COMPUTE NUM POINTS FOR MOVING DYNAMIC THRESHOLDS

    import numpy as np
    from tqdm import tqdm

    grid_cell_size = 0.109375
    #dataset_path = "/home/pokorsi1/data/waymo_flow/preprocess"
    dataset_path = "../../data/waymoflow/"

    data_module = WaymoSeqDataModule(
        dataset_directory=dataset_path,
        grid_cell_size=grid_cell_size,
        x_min=-35,
        x_max=35,
        y_min=-35,
        y_max=35,
        z_min=0.45,
        z_max=10,
        batch_size=1,
        has_test=False,
        num_workers=0,
        n_pillars_x=640,
        n_points=None,
        apply_pillarization=True)

    data_module.setup()
    train_dl = data_module.train_dataloader()

    SUM = np.array([0]).astype('Q')
    SUM_min = np.array([0]).astype('Q')

    for pcli, Ti in tqdm(train_dl):
        # Create pcl from features vector
        #num_points = x[0][0].shape[1]
        (t3_to_t0, t3_to_t1, t3_to_t2) = Ti
        del Ti

        (t0_frame, t1_frame, t2_frame, t3_frame) = pcli
        del pcli

        a = None
