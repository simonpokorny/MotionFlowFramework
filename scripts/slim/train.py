from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentTypeError
from configs.utils import load_config
from models.SLIM import SLIM

from datasets.waymoflow import WaymoDataModule
from datasets.kitti import KittiDataModule

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
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--gpus', default=1, type=int, help="GPU parameter of PL Trainer class.")
    parser.add_argument('--accelerator', default="gpu", type=str,
                        help="Accelerator to use. Set to ddp for multi GPU training.")  # Param of Trainer


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config("../../configs/slim.yaml")
    data_cfg = cfg["data"]
    del cfg["data"]
    model = SLIM(config=cfg)




    grid_cell_size = (data_cfg["x_max"] + abs(data_cfg["x_min"])) / data_cfg["grid_size"]

    n_pillars_x = data_cfg["n_pillars_x"]
    n_pillars_y = data_cfg["n_pillars_x"]

    apply_pillarization = data_cfg["apply_pillarization"]




    if args.dataset == 'waymo':

        data_module = WaymoDataModule(dataset_path, grid_cell_size=grid_cell_size, x_min=args.x_min,
                                      x_max=args.x_max, y_min=args.y_min,
                                      y_max=args.y_max, z_min=args.z_min, z_max=args.z_max,
                                      batch_size=args.batch_size,
                                      has_test=args.test_data_available,
                                      num_workers=args.num_workers,
                                      n_pillars_x=n_pillars_x,
                                      n_points=args.n_points, apply_pillarization=apply_pillarization)

        from data.util import ApplyPillarization
        pilarization = ApplyPillarization(grid_cell_size=grid_cell_size, x_min=args.x_min, y_min=args.y_min,
                           z_min=args.z_min, z_max=args.z_max, n_pillars_x=n_pillars_x,
                           )


    elif args.dataset == 'kitti':
        data_module = KittiDataModule(dataset_path,
                                batch_size=args.batch_size,
                                has_test=args.test_data_available,
                                num_workers=args.num_workers,
                                n_points=args.n_points)
    else:
        raise ValueError('Dataset {} not available'.format(args.dataset))




    gradient_batch_acc = 1  # Do not accumulate batches before performing optimizer step
    if args.full_batch_size is not None:
        gradient_batch_acc = int(args.full_batch_size / args.batch_size)
        print(f"A full batch size is specified. The model will perform gradient update after {gradient_batch_acc} "
              f"smaller batches of size {args.batch_size} to approx. total batch size of {args.full_batch_size}."
              f"PLEASE NOTE that if the network includes layers that need larger batch sizes such as BatchNorm "
              f"they are still computed for each forward pass.")

    plugins = None
    if args.disable_ddp_unused_check:
        if not args.accelerator == "ddp":
            print("FATAL: DDP unused checks can only be disabled when DDP is used as accelerator!")
            exit(1)
        print("Disabling unused parameter check for DDP")
        # plugins = DDPPlugin(find_unused_parameters=False)

    # Add a callback for checkpointing after each epoch and the model with best validation loss
    # checkpoint_callback = ModelCheckpoint(monitor="val/loss", mode="min", save_last=True)

    # Max epochs can be configured here to, early stopping is also configurable.
    # Some things are definable as callback from pytorch_lightning.callback
    # trainer = pl.Trainer.from_argparse_args(args,
    #                                         precision=32,  # Precision 16 does not seem to work with batchNorm1D
    #                                         gpus=args.gpus if torch.cuda.is_available() else 0,  # -1 means "all GPUs"
    #                                         logger=logger,
    #                                         accumulate_grad_batches=gradient_batch_acc,
    #                                         log_every_n_steps=5,
    #                                         plugins=plugins,
    #                                         callbacks=[checkpoint_callback]
    #                                         )  # Add Trainer hparams if desired
    # The actual train loop
    # trainer.fit(model, data_module)

    # Run also the testing
    # if args.test_data_available and not args.fast_dev_run:
    #     trainer.test()  # Also loads the best checkpoint automatically

    # dummy data for testing
