import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentTypeError
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

sys.path.append('../../')

from callbacks import SaveViz
from configs import load_config
from datasets import NuScenesDataModule, WaymoDataModule, KittiDataModule
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
                        choices=["waymo", 'rawkitti', 'nuscenes'],
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
    import warnings

    warnings.filterwarnings("ignore")

    args = parse_args()
    EXPERIMENT_PATH = Path("experiments")
    os.makedirs(EXPERIMENT_PATH, exist_ok=True)

    # Loading config
    cfg = load_config("../../configs/slim.yaml")

    # Creating the model
    model = SLIM(config=cfg, dataset=args.dataset)

    if args.resume_from_checkpoint is not None:
        raise NotImplementedError()

    if args.dataset == 'waymo':
        dataset_path = args.data_path if args.data_path is not None else "../../data/waymoflow_subset"
        data_cfg = cfg["data"][args.dataset]
        grid_cell_size = (data_cfg["x_max"] + abs(data_cfg["x_min"])) / data_cfg["n_pillars_x"]
        data_module = WaymoDataModule(dataset_directory=dataset_path, grid_cell_size=grid_cell_size, **data_cfg)

    elif args.dataset == 'rawkitti':
        dataset_path = args.data_path if args.data_path is not None else "/home/pokorsi1/data/rawkitti/prepared"
        data_cfg = cfg["data"][args.dataset]
        grid_cell_size = (data_cfg["x_max"] + abs(data_cfg["x_min"])) / data_cfg["n_pillars_x"]
        data_module = KittiDataModule(dataset_directory=dataset_path, grid_cell_size=grid_cell_size, **data_cfg)

    elif args.dataset == "nuscenes":
        dataset_path = args.data_path if args.data_path is not None else "/home/pokorsi1/data/nuscenes/preprocess"
        data_cfg = cfg["data"][args.dataset]
        grid_cell_size = (data_cfg["x_max"] + abs(data_cfg["x_min"])) / data_cfg["n_pillars_x"]
        data_module = NuScenesDataModule(dataset_directory=dataset_path, grid_cell_size=grid_cell_size, **data_cfg)
    else:
        raise ValueError('Dataset {} not available yet'.format(args.dataset))

    try:
        version = len(os.listdir(os.path.join(EXPERIMENT_PATH, args.dataset, "lightning_logs")))
    except:
        version = 0

    print(f"Saved under version num : {version}")

    callbacks = [ModelCheckpoint(dirpath=EXPERIMENT_PATH / args.dataset / "checkpoints" / f"version_{version}",
                                 save_weights_only=True, every_n_train_steps=1000, save_last=True, save_top_k=-1)]

                # SaveViz(dirpath=EXPERIMENT_PATH / args.dataset / "visualization" / f"version_{version}",
                #         every_n_train_steps=1000)]

    loggers = [TensorBoardLogger(save_dir=EXPERIMENT_PATH, name=f"{args.dataset}/lightning_logs",
                                 log_graph=True, version=version),
               CSVLogger(save_dir=EXPERIMENT_PATH, name=f"{args.dataset}/lightning_logs", version=version)]

    # trainer with no validation loop
    trainer = pl.Trainer(limit_val_batches=0, num_sanity_val_steps=0, devices=1, accelerator=args.accelerator,
                         enable_checkpointing=True, fast_dev_run=args.fast_dev_run, max_epochs=10,
                         logger=loggers, callbacks=callbacks)  # , limit_train_batches=1)

    # Trainer where we will validete every 1000 iters (used as test during training)
    # trainer = pl.Trainer(val_check_interval=1000, devices=1, accelerator=args.accelerator, enable_checkpointing=True,
    #                     fast_dev_run=args.fast_dev_run, max_epochs=3,
    #                     logger=loggers, callbacks=callbacks)  # , limit_train_batches=1)


    trainer.fit(model, data_module)
