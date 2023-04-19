import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentTypeError
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from callbacks import SaveViz

sys.path.append('../../')

from configs import load_config
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

    parser.add_argument('--resume_from_checkpoint', type=str, required=True,
                        help="Path to ckpt file to resume from. Parameter from PytorchLightning Trainer.")
    # Data related arguments
    parser.add_argument('--dataset', choices=["waymo", 'rawkitti', 'nuscenes', 'kittisf'],
                        help="Dataset Type to train on.", required=True)
    parser.add_argument('--dataset_trained_on', choices=["waymo", 'rawkitti', 'nuscenes'],
                        help="Dataset Type to train on.", required=True)
    parser.add_argument('--data_path', default=None, type=str, help="Specify the data dir.")
    parser.add_argument('--fast_dev_run', type=str2bool, nargs='?', const=True, default=False,
                        help="If fast_dev_run is true (train is in developer mode for one batch)")
    parser.add_argument('--gpus', default=1, type=int, help="GPU parameter of PL Trainer class.")
    parser.add_argument('--accelerator', default="cpu", type=str,
                        help="Accelerator to use. Set to ddp for multi GPU training.")  # Param of Trainer

    args = parser.parse_args()
    return args


def get_datamodule(dataset: str, data_path: str):
    data_cfg = cfg["data"][dataset]
    grid_cell_size = (data_cfg["x_max"] + abs(data_cfg["x_min"])) / data_cfg["n_pillars_x"]

    if dataset == 'waymo':
        dataset_path = data_path if data_path is not None else "../../data/waymoflow_subset"
        data_module = WaymoDataModule(dataset_directory=dataset_path, grid_cell_size=grid_cell_size, **data_cfg)
    elif dataset == 'rawkitti':
        dataset_path = data_path if data_path is not None else "/home/pokorsi1/data/rawkitti/prepared"
        data_module = KittiDataModule(dataset_directory=dataset_path, grid_cell_size=grid_cell_size, **data_cfg)
    elif args.dataset == "kittisf":
        dataset_path = data_path if data_path is not None else "../data/kittisf/"
        data_module = KittiSceneFlowDataModule(dataset_directory=dataset_path, grid_cell_size=grid_cell_size,
                                               **data_cfg)
    elif args.dataset == "nuscenes":
        dataset_path = data_path if data_path is not None else "/home/pokorsi1/data/nuscenes/preprocess"
        data_module = NuScenesDataModule(dataset_directory=dataset_path, grid_cell_size=grid_cell_size, **data_cfg)
    else:
        raise ValueError('Dataset {} not available yet'.format(args.dataset))
    return data_module


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    args = parse_args()
    EXPERIMENT_PATH = Path("eval")
    os.makedirs(EXPERIMENT_PATH, exist_ok=True)

    # Loading config
    cfg = load_config("../../configs/slim.yaml")

    # Creating and loading the model and datamodule
    model = SLIM(config=cfg, dataset=args.dataset_trained_on)
    model = model.load_from_checkpoint(args.resume_from_checkpoint)
    data_module = get_datamodule(args.dataset, args.data_path)

    try:
        version = len(os.listdir(os.path.join(EXPERIMENT_PATH, args.dataset, "lightning_logs")))
    except:
        version = 0
    print(f"Saved under version num : {version}")

    callbacks = [SaveViz(dirpath=EXPERIMENT_PATH / args.dataset / "visualization" / f"version_{version}",
                         every_n_test_steps=1)]

    loggers = [TensorBoardLogger(save_dir=EXPERIMENT_PATH, name=f"{args.dataset}/lightning_logs",
                                 log_graph=True, version=version),
               CSVLogger(save_dir=EXPERIMENT_PATH, name=f"{args.dataset}/lightning_logs", version=version)]

    # trainer with no validation loop
    trainer = pl.Trainer(limit_val_batches=0, num_sanity_val_steps=0, devices=1, accelerator=args.accelerator,
                         enable_checkpointing=True, fast_dev_run=args.fast_dev_run, max_epochs=1,
                         logger=loggers, callbacks=callbacks)

    trainer.test(model, data_module)
    print("done")
