### Implemented dataset for motion flow 

- Raw Kitti
- Kitti Scene Flow
- NuScenes
- Waymo

### Adding a new dataset



For correct usage, the data should be structured in folder, where is train, val and optionaly test folder. 
For new dataset class it is neccessery to write new class, which inherit from BaseDataset. 

Following functions should be written. 


```python
class CustomDataset(BaseDataset):
    def __init__(self, data_path,
             drop_invalid_point_function=None,
             point_cloud_transform=None,
             n_points=None,
             apply_pillarization=True):
        super().__init__()

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
            t0_frame: pointcloud in shape [1, N, features]
            t1_frame: pointcloud in shape [1, N, features]
        """

        raise NotImplementedError()

    def _get_pose_transform(self, index):
        """
        For each dataset should be separetly written. Returns transforamtion from t0 to t1
        Returns:
            t0_to_t1: in shape [1, 4, 4]
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
        Optional. For each dataset should be separetly written. Returns gt flow in shape [1, N, channels].
        """
        return None
```
If dataset do not contain ground-truth flow or something else, just rewrite the function which the dataset contain. 


For datamodule it is neccessary to write this class with correct name and put to super().__init__ the correct dataset.

```python
class CustomDataModule(BaseDataModule):
    def __init__(self,
                 dataset_directory: str = "~/data/customdataset/prepared/",
                 # These parameters are specific to the dataset
                 grid_cell_size: float = 0.109375,
                 x_min: float = -35.,
                 x_max: float = 35.,
                 y_min: float = -35.,
                 y_max: float = 35,
                 z_min: float = -10,
                 z_max: float = 10,
                 n_pillars_x: int = 640,
                 n_pillars_y: int = 640,
                 batch_size: int = 1,
                 has_test=False,
                 num_workers=1,
                 n_points=None,
                 apply_pillarization=True,
                 shuffle_train=True,
                 point_features=7):

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
```
