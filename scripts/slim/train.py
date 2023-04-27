import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentTypeError
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

sys.path.append('../../')

from configs import get_config
from datasets import get_datamodule
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
                        choices=["waymo", 'rawkitti', 'nuscenes', 'petr_dataset'],
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
    cfg = get_config("../../configs/slim.yaml", dataset=args.dataset)

    # Creating the model
    model = SLIM(config=cfg, dataset=args.dataset)

    # Load from checkpoint
    if args.resume_from_checkpoint is not None:
        raise NotImplementedError()

    # model = model.load_from_checkpoint("/home/pokorsi1/motion_learning/scripts/slim/experiments/nuscenes/checkpoints/version_1/epoch=0-step=8000.ckpt")
    # model = model.load_from_checkpoint("/home/pokorsi1/motion_learning/scripts/slim/experiments/waymo/checkpoints/version_3/last.ckpt")

    # Get datamodule
    data_module = get_datamodule(dataset=args.dataset, data_path=args.data_path, cfg=cfg)

    try:
        version = len(os.listdir(os.path.join(EXPERIMENT_PATH, args.dataset, "lightning_logs")))
    except:
        version = 0

    print(f"Saved under version num : {version}")

    callbacks = [ModelCheckpoint(dirpath=EXPERIMENT_PATH / args.dataset / "checkpoints" / f"version_{version}",
                                 save_weights_only=False, every_n_train_steps=1000, save_last=True, save_top_k=-1)]

    # SaveViz(dirpath=EXPERIMENT_PATH / args.dataset / "visualization" / f"version_{version}",
    #         every_n_train_steps=1000)]

    loggers = [TensorBoardLogger(save_dir=EXPERIMENT_PATH, name=f"{args.dataset}/lightning_logs",
                                 log_graph=True, version=version),
               CSVLogger(save_dir=EXPERIMENT_PATH, name=f"{args.dataset}/lightning_logs", version=version)]

    # trainer with no validation loop
    trainer = pl.Trainer(limit_val_batches=0, num_sanity_val_steps=0, devices=1, accelerator=args.accelerator,
                         enable_checkpointing=True, fast_dev_run=args.fast_dev_run, max_epochs=10,
                         logger=loggers, callbacks=callbacks)

    trainer.fit(model, data_module)
