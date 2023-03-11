import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentTypeError

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

import sys
sys.path.append('../../')

from callbacks import SaveInference
from configs import load_config
from datasets.kitti import KittiDataModule
from datasets.waymoflow import WaymoDataModule
from models.SLIM import SLIM


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def parse_args():
    """
    Setup all arguments and parse them from commandline.
    :return: The ArgParser args object with everything parsed.
    """
    parser = ArgumentParser(description="Training script for FastFlowNet and FlowNet3D "
                                        "based on Waymo or flying thing dataset",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--resume_from_checkpoint', type=str,
                        help="Path to ckpt file to resume from. Parameter from PytorchLightning Trainer.")
    # Data related arguments
    parser.add_argument('--dataset', default='waymo',
                        choices=["waymo", 'rawkitti'],
                        help="Dataset Type to train on.")
    parser.add_argument('--data_path', default=None, type=str, help="Specify the data dir.")

    parser.add_argument('--fast_dev_run', type=str2bool, nargs='?', const=True, default=False,
                        help="If fast_dev_run is true (train is in developer mode for one batch)")
    parser.add_argument('--gpus', default=1, type=int, help="GPU parameter of PL Trainer class.")
    parser.add_argument('--accelerator', default="cpu", type=str,
                        help="Accelerator to use. Set to ddp for multi GPU training.")  # Param of Trainer

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    EXPERIMENT_PATH = "experiments"
    os.makedirs(EXPERIMENT_PATH, exist_ok=True)

    args = parse_args()
    cfg = load_config("../../configs/slim.yaml")
    data_cfg = cfg["data"]
    del cfg["data"]
    model = SLIM(config=cfg)

    # args.dataset = "kitti"

    if args.resume_from_checkpoint is not None:
        raise NotImplementedError()

    grid_cell_size = (data_cfg["x_max"] + abs(data_cfg["x_min"])) / data_cfg["n_pillars_x"]
    if args.dataset == 'waymo':
        dataset_path = args.data_path if args.data_path is not None else "../../data/waymoflow_subset"
        data_module = WaymoDataModule(dataset_directory=dataset_path, grid_cell_size=grid_cell_size,
                                      x_min=data_cfg["x_min"], x_max=data_cfg["x_max"],
                                      y_min=data_cfg["y_min"], y_max=data_cfg["y_max"],
                                      z_min=data_cfg["z_min"], z_max=data_cfg["z_max"],
                                      batch_size=data_cfg["batch_size"],
                                      has_test=False,
                                      num_workers=data_cfg["num_workers"],
                                      n_pillars_x=data_cfg["n_pillars_x"],
                                      n_points=data_cfg["n_points"],
                                      apply_pillarization=data_cfg["apply_pillarization"])

    elif args.dataset == 'rawkitti':
        dataset_path = args.data_path if args.data_path is not None else "/home/pokorsi1/data/rawkitti/prepared"
        data_module = KittiDataModule(dataset_directory=dataset_path, grid_cell_size=grid_cell_size,
                                      x_min=data_cfg["x_min"], x_max=data_cfg["x_max"],
                                      y_min=data_cfg["y_min"], y_max=data_cfg["y_max"],
                                      z_min=data_cfg["z_min"], z_max=data_cfg["z_max"],
                                      batch_size=data_cfg["batch_size"],
                                      has_test=False,
                                      num_workers=data_cfg["num_workers"],
                                      n_pillars_x=data_cfg["n_pillars_x"],
                                      n_points=data_cfg["n_points"],
                                      apply_pillarization=data_cfg["apply_pillarization"])
    else:
        raise ValueError('Dataset {} not available yet'.format(data_cfg["dataset"]))

    try:
        version = len(os.listdir(os.path.join(EXPERIMENT_PATH, "lightning_logs")))
    except:
        version = 0

    callbacks = [ModelCheckpoint(dirpath=EXPERIMENT_PATH, save_weights_only=True, every_n_train_steps=1000),
                 SaveInference(dirpath=EXPERIMENT_PATH, name="lightning_logs", version=version)]
    loggers = [TensorBoardLogger(save_dir=EXPERIMENT_PATH, name="lightning_logs", log_graph=True, version=version),
               CSVLogger(save_dir=EXPERIMENT_PATH, name="lightning_logs", version=version)]

    # trainer with no validation loop
    trainer = pl.Trainer(limit_val_batches=0, num_sanity_val_steps=0, devices=1, accelerator=args.accelerator,
                         enable_checkpointing=True, fast_dev_run=args.fast_dev_run, max_epochs=50)

    trainer.fit(model, data_module)
