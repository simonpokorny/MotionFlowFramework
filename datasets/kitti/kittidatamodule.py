from datasets.base import BaseDataModule
from datasets.kitti.kittidataset import KittiRawDataset


class KittiDataModule(BaseDataModule):
    def __init__(self,
                 dataset_directory: str = "~/data/rawkitti/prepared/",
                 # These parameters are specific to the dataset
                 grid_cell_size: float = 0.109375,
                 x_min: float = -35.,
                 x_max: float = 35.,
                 y_min: float = -35.,
                 y_max: float = 35,
                 z_min: float = -10,
                 z_max: float = 10,
                 n_pillars_x: int = 640,
                 batch_size: int = 1,
                 has_test=False,
                 num_workers=1,
                 n_points=None,
                 apply_pillarization=True,
                 shuffle_train=True):
        """
        This class defines a PyTorch Lightning DataModule that loads and preprocesses data from a specified dataset
        directory using the provided arguments. For correct using, it is neccesery to implement a class dataset for
        specific data based on BasaDataset.

        Args:
            dataset (class): A PyTorch Dataset class that inherits from `BaseDataset`.
            dataset_directory (str): Path to the directory containing the train, val and test dataset.
            grid_cell_size (float): The size of the grid cell for the pillarization step.
            x_min (float): The minimum x-coordinate which will be used.
            x_max (float): The maximum x-coordinate which will be used.
            y_min (float): The minimum y-coordinate which will be used.
            y_max (float): The maximum y-coordinate which will be used.
            z_min (float): The minimum z-coordinate which will be used.
            z_max (float): The maximum z-coordinate which will be used.
            n_pillars_x (int): The number of pillars along the x-axis for the pillarization step.
            batch_size (int, optional): The batch size to use for the DataLoader (default: 32).
            has_test (bool, optional): Whether the dataset has a test set (default: False).
            num_workers (int, optional): The number of worker processes to use for loading the data (default: 1).
            n_points (int, optional): The number of points to sample from each point cloud (default: None, which means to use all points).
            apply_pillarization (bool, optional): Whether to apply pillarization to the point clouds (default: True).
            shuffle_train (bool, optional): Whether to shuffle the training set (default: True).

        """
        super().__init__(dataset=KittiRawDataset,
                         dataset_directory=dataset_directory,
                         grid_cell_size=grid_cell_size,
                         x_min=x_min,
                         x_max=x_max,
                         y_min=y_min,
                         y_max=y_max,
                         z_min=z_min,
                         z_max=z_max,
                         n_pillars_x=n_pillars_x,
                         batch_size=batch_size,
                         has_test=has_test,
                         num_workers=num_workers,
                         n_points=n_points,
                         apply_pillarization=apply_pillarization,
                         shuffle_train=shuffle_train)


if __name__ == "__main__":
    datamodule = KittiDataModule(dataset_directory="../../data/rawkitti/", num_workers=0)
    datamodule.setup("fit")
    train_dl = datamodule.train_dataloader()
    train_ds = datamodule._train_
    a = train_ds[0]
    for bs in train_dl:

        tmp = None

