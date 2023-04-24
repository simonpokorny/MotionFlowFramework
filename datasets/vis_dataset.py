if __name__ == "__main__":
    DATASET = "waymo"
    assert DATASET in ["waymo", "rawkitti", "kittisf", "nuscenes"]

    from tqdm import tqdm
    import open3d as o3d

    from configs import load_config
    from visualization.plot import show_flow, save_trans_pcl
    from datasets import KittiDataModule, WaymoDataModule, KittiSceneFlowDataModule, NuScenesDataModule

    cfg = load_config("../configs/slim.yaml")

    data_cfg = cfg["data"][DATASET]
    grid_cell_size = (data_cfg["x_max"] + abs(data_cfg["x_min"])) / data_cfg["n_pillars_x"]
    data_cfg["num_workers"] = 0

    if DATASET == 'waymo':
        dataset_path = "../data/waymoflow"
        data_module = WaymoDataModule(dataset_directory=dataset_path, grid_cell_size=grid_cell_size, **data_cfg)
    elif DATASET == 'rawkitti':
        dataset_path = "../data/rawkitti/"
        data_module = KittiDataModule(dataset_directory=dataset_path, grid_cell_size=grid_cell_size, **data_cfg)
    elif DATASET == "kittisf":
        dataset_path = "../data/kittisf/"
        data_module = KittiSceneFlowDataModule(dataset_directory=dataset_path, grid_cell_size=grid_cell_size,
                                               **data_cfg)
    elif DATASET == "nuscenes":
        dataset_path = "../data/nuscenes"
        data_module = NuScenesDataModule(dataset_directory=dataset_path, grid_cell_size=grid_cell_size, **data_cfg)
    else:
        raise ValueError('Dataset {} not available yet'.format(DATASET))

    data_module.setup()
    if DATASET != "kittisf":
        train_dl = data_module.train_dataloader()
    else:
        train_dl = data_module.test_dataloader()

    #train_dl = data_module.test_dataloader()

    for x, flow, T_gt in tqdm(train_dl):
        # Create pcl from features vector
        pc_previous = x[0][0]
        pc_current = x[1][0]
        flow = flow[:, :, :3]

        pc_previous_numpy = (pc_previous[0, :, :3] + pc_previous[0, :, 3:6]).detach().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_previous_numpy)

        # Save PointCloud as PLY file
        o3d.io.write_point_cloud("pcl.ply", pcd)

        #save_trans_pcl(T_gt, pc_previous, pc_current, " ", "synchronized_pcl", show=True)
        # T_gt = torch.linalg.inv(T_gt)
        # save_trans_pcl(T_gt, pc_previous, pc_current, " ", "synchronized_pcl", show=True)

        show_flow(pcl=pc_previous, flow=flow, pcl2=pc_current)
