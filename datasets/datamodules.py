from datasets import WaymoDataModule, KittiDataModule, KittiSceneFlowDataModule, NuScenesDataModule, PetrDataModule


def get_datamodule(dataset: str, data_path: str, cfg: dict):
    data_cfg = cfg["data"][dataset]
    grid_cell_size = (data_cfg["x_max"] + abs(data_cfg["x_min"])) / data_cfg["n_pillars_x"]

    if dataset == 'waymo':
        dataset_path = data_path if data_path is not None else "../../data/waymoflow_subset"
        data_cfg["has_test"] = True
        data_module = WaymoDataModule(dataset_directory=dataset_path, grid_cell_size=grid_cell_size, **data_cfg)
    elif dataset == 'rawkitti':
        dataset_path = data_path if data_path is not None else "/home/pokorsi1/data/rawkitti/prepared"
        data_module = KittiDataModule(dataset_directory=dataset_path, grid_cell_size=grid_cell_size, **data_cfg)
    elif dataset == "kittisf":
        dataset_path = data_path if data_path is not None else "../data/kittisf/"
        data_module = KittiSceneFlowDataModule(dataset_directory=dataset_path, grid_cell_size=grid_cell_size,
                                               **data_cfg)
    elif dataset == "nuscenes":
        dataset_path = data_path if data_path is not None else "/home/pokorsi1/data/nuscenes/preprocess"
        data_module = NuScenesDataModule(dataset_directory=dataset_path, grid_cell_size=grid_cell_size, **data_cfg)
    elif dataset == "petr_dataset":
        dataset_path = data_path
        grid_cell_size = (data_cfg["x_max"] + abs(data_cfg["x_min"])) / data_cfg["n_pillars_x"]
        data_module = PetrDataModule(dataset_directory=dataset_path, grid_cell_size=grid_cell_size, **data_cfg)
    else:
        raise ValueError('Dataset {} not available yet'.format(dataset))
    return data_module
