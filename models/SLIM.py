import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from configs.utils import load_config
from losses.flow import rigid_cycle_loss, NN_loss, smoothness_loss, static_point_loss
from models.networks import PillarFeatureNetScatter, PointFeatureNet, MovingAverageThreshold, RAFT
from models.networks.slimdecoder import OutputDecoder
from models.utils import init_weights


class SLIM(pl.LightningModule):
    def __init__(self, config, dataset):
        """
        Args:
            config (dict): Config is based on configs from configs/slim.yaml
            dataset (str): type of the dataset
        """
        super(SLIM, self).__init__()
        self.save_hyperparameters()  # Store the constructor parameters into self.hparams
        assert type(config) == dict
        self.config = config

        # just only example input to model (useful for tensorboard to make computional graph)
        self.n_features = config["data"][dataset]["point_features"]
        t1 = [torch.rand([1, 20, self.n_features]), torch.randint(0, 20, (1, 20)),
              torch.ones((1, 20), dtype=torch.bool)]
        t0 = [torch.rand([1, 20, self.n_features]), torch.randint(0, 20, (1, 20)),
              torch.ones((1, 20), dtype=torch.bool)]
        transf = torch.eye(4)[None, :]
        # self.example_input_array = (t0, t1), transf

        self.last_output = None
        self._loss_BCE = torch.nn.BCELoss()

        self.save_hyperparameters()  # Store the constructor parameters into self.hparams
        self.n_pillars_x = config["data"][dataset]["n_pillars_x"]
        self.n_pillars_y = config["data"][dataset]["n_pillars_y"]

        self._point_feature_net = PointFeatureNet(in_features=self.n_features, out_features=64)
        self._point_feature_net.apply(init_weights)

        self._pillar_feature_net = PillarFeatureNetScatter(n_pillars_x=self.n_pillars_x, n_pillars_y=self.n_pillars_y)
        self._pillar_feature_net.apply(init_weights)

        # Raft init weights is done internally.
        self._raft = RAFT(**config["raft"])
        # self._raft.apply(init_weights)

        self._moving_dynamicness_threshold = MovingAverageThreshold(**config["moving_threshold"][dataset])

        self._decoder_fw = OutputDecoder(**config["decoder"])
        self._decoder_bw = OutputDecoder(**config["decoder"])

    def _transform_point_cloud_to_embeddings(self, pc, mask):
        """
         A method that takes a point cloud and a mask and returns the corresponding embeddings.
         The method flattens the point cloud and mask, applies the point feature network,
         and then reshapes the embeddings to their original dimensions.
        """
        pc_flattened = pc.flatten(0, 1)
        mask_flattened = mask.flatten(0, 1)
        # Init the result tensor for our data. This is necessary because the point net
        # has a batch norm and this needs to ignore the masked points
        batch_pc_embedding = torch.zeros((pc_flattened.size(0), 64), device=pc.device, dtype=pc.dtype)
        # Flatten the first two dimensions to get the points as batch dimension
        batch_pc_embedding[mask_flattened] = self._point_feature_net(pc_flattened[mask_flattened])
        # This allows backprop towards the MLP: Checked with backward hooks. Gradient is present.
        # Output is (batch_size * points, embedding_features)
        # Retransform into batch dimension (batch_size, max_points, embedding_features)
        batch_pc_embedding = batch_pc_embedding.unflatten(0, (pc.size(0), pc.size(1)))
        # 241.307 MiB    234
        return batch_pc_embedding

    def _batch_grid_2D(self, batch_grid):
        """
        A method that takes a batch of grid indices and returns the corresponding 2D grid coordinates.
        The method calculates the x and y indices of the grid points using the number of pillars in
        the x and y dimensions, respectively, and then concatenates them along the second dimension.
        """
        # Numpy version
        # grid = np.hstack(
        #    ((batch_grid // self.n_pillars_x)[:, np.newaxis], (batch_grid % self.n_pillars_y)[:, np.newaxis]))
        # grid = np.moveaxis(grid, -1, 1)

        # Pytorch version
        grid = torch.cat(((batch_grid // self.n_pillars_x).unsqueeze(1),
                          (batch_grid % self.n_pillars_y).unsqueeze(1)), dim=1)
        return grid.transpose(-1, 1)  # Equivalent to np.moveaxis(grid, -1, 1)

    def _filled_pillar_mask(self, batch_grid, batch_mask):
        """
        A method that takes a batch of grid indices and masks and returns a tensor with a 1 in the location
        of each grid point and a 0 elsewhere. The method creates a tensor of zeros with the same shape as
        the voxel grid, and then sets the locations corresponding to the grid points in the batch to 1.
        """
        bs = batch_grid.shape[0]
        # pillar mask
        pillar_mask = torch.zeros((bs, 1, self.n_pillars_x, self.n_pillars_y), device=batch_grid.device)
        #
        x = batch_grid[batch_mask][..., 0]
        y = batch_grid[batch_mask][..., 1]
        pillar_mask[:, :, x, y] = 1
        return pillar_mask

    def forward(self, x, transforms_matrices):
        """
        The usual forward pass function of a torch module
        :param x:
        :param transforms_matrices:
        :return:
        """
        # 1. Do scene encoding of each point cloud to get the grid with pillar embeddings
        # Input is a point cloud each with shape (N_points, point_features)

        # The input here is more complex as we deal with a batch of point clouds
        # that do not have a fixed amount of points
        # x is a tuple of two lists representing the batches
        previous_batch, current_batch = x
        # trans is a tensor representing transforms from previous to current frame (t0 -> t1)

        P_T_C = transforms_matrices.type(self.dtype)
        C_T_P = torch.linalg.inv(P_T_C).type(self.dtype)

        previous_batch_pc, previous_batch_grid, previous_batch_mask = previous_batch
        current_batch_pc, current_batch_grid, current_batch_mask = current_batch

        # Cut embedings to wanted number of features
        previous_batch_pc = previous_batch_pc[:, :, :self.n_features]
        current_batch_pc = current_batch_pc[:, :, :self.n_features]

        # For some reason the datatype of the input is not changed to correct precision
        previous_batch_pc = previous_batch_pc.type(self.dtype)
        current_batch_pc = current_batch_pc.type(self.dtype)

        # pointwise_voxel_coordinates_fs is in shape [BS, num points, 2],
        # where last dimension belong to location of point in voxel grid
        current_voxel_coordinates = self._batch_grid_2D(current_batch_grid)
        previous_voxel_coordinates = self._batch_grid_2D(previous_batch_grid)

        # Create bool map of filled/non-filled pillars
        current_batch_pillar_mask = self._filled_pillar_mask(current_voxel_coordinates, current_batch_mask)
        previous_batch_pillar_mask = self._filled_pillar_mask(previous_voxel_coordinates, previous_batch_mask)

        # batch_pc = (batch_size, N, 8) | batch_grid = (n_batch, N, 2) | batch_mask = (n_batch, N)
        # The grid indices are (batch_size, max_points) long. But we need them as
        # (batch_size, max_points, feature_dims) to work. Features are in all necessary cases 64.
        # Expand does only create multiple views on the same datapoint and not allocate extra memory
        current_batch_grid = current_batch_grid.unsqueeze(-1).expand(-1, -1, 64)
        previous_batch_grid = previous_batch_grid.unsqueeze(-1).expand(-1, -1, 64)

        # Pass the whole batch of point clouds to get the embedding for each point in the cloud
        # Input pc is (batch_size, max_n_points, features_in)
        # per each point, there are 8 features: [cx, cy, cz,  Δx, Δy, Δz, l0, l1], as stated in the paper
        previous_batch_pc_embedding = self._transform_point_cloud_to_embeddings(previous_batch_pc,
                                                                                previous_batch_mask).type(self.dtype)
        # previous_batch_pc_embedding = [n_batch, N, 64]
        # Output pc is (batch_size, max_n_points, embedding_features)
        current_batch_pc_embedding = self._transform_point_cloud_to_embeddings(current_batch_pc,
                                                                               current_batch_mask).type(self.dtype)

        # Now we need to scatter the points into their 2D matrix
        # batch_pc_embeddings -> (batch_size, N, 64)
        # batch_grid -> (batch_size, N, 64)
        # No learnable params in this part
        previous_pillar_embeddings = self._pillar_feature_net(previous_batch_pc_embedding, previous_batch_grid)
        current_pillar_embeddings = self._pillar_feature_net(current_batch_pc_embedding, current_batch_grid)
        # pillar_embeddings = (batch_size, 64, 640, 640)

        # Concatenate the previous and current batches along a new dimension.
        # This allows to have twice the amount of entries in the forward pass
        # of the encoder which is good for batch norm.
        pillar_embeddings = torch.stack((previous_pillar_embeddings, current_pillar_embeddings), dim=1)
        # This is now (batch_size, 2, 64, 640, 640) large
        pillar_embeddings = pillar_embeddings.flatten(0, 1)
        # Flatten into (batch_size * 2, 64, 512, 512) for encoder forward pass.

        # The grid map is ready in shape (BS, 64, 640, 640)

        # 2. RAFT Encoder with motion flow backbone
        # Output for forward pass and backward pass
        # Each of the output is a list of num_iters x [1, 9, n_pillars_x, n_pillars_x]
        # logits, static_flow, dynamic_flow, weights are concatinate in channels in shapes [4, 2, 2, 1]
        outputs_fw, outputs_bw = self._raft(pillar_embeddings)

        # Transformation matrix Current (t1) to Previous (t0)
        # C_T_P = torch.linalg.inv(G_T_C) @ G_T_P
        # Transformation matrix Previous (t0) to Current (t1)
        # P_T_C = torch.linalg.inv(G_T_P) @ G_T_C

        predictions_fw = []
        predictions_bw = []

        for it, (raft_output_0_1, raft_output_1_0) in enumerate(zip(outputs_fw, outputs_bw)):
            prediction_fw = self._decoder_fw(
                network_output=raft_output_0_1,
                dynamicness_threshold=self._moving_dynamicness_threshold.value().to(self.device),
                pc=previous_batch_pc,
                pointwise_voxel_coordinates_fs=previous_voxel_coordinates,
                pointwise_valid_mask=previous_batch_mask,
                filled_pillar_mask=previous_batch_pillar_mask.type(torch.bool),
                odom=P_T_C,  # TODO má tam být P_T_C ?
                inv_odom=C_T_P,
                it=it)

            prediction_bw = self._decoder_bw(
                network_output=raft_output_1_0,
                dynamicness_threshold=self._moving_dynamicness_threshold.value().to(self.device),
                pc=current_batch_pc,
                pointwise_voxel_coordinates_fs=current_voxel_coordinates,
                pointwise_valid_mask=current_batch_mask,
                filled_pillar_mask=current_batch_pillar_mask.type(torch.bool),
                odom=C_T_P,
                inv_odom=P_T_C,
                it=it)

            predictions_fw.append(prediction_fw)
            predictions_bw.append(prediction_bw)
        return predictions_fw, predictions_bw, previous_batch_pc, current_batch_pc

    def configure_optimizers(self):
        """
        Also pytorch lightning specific.
        Define the optimizers in here this will return the optimizer and schedulars that is used to train this module.
        :return: The optimizer and schedular to use

        SLIM official setup:

        initial: 0.0001
        step_decay:
        decay_ratio: 0.5
        step_length: 60000

        warm_up:
        initial: 0.01
        step_length: 2000

        """
        self.lr = 0.0001
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)

        decay = lambda step: 0.5 ** int(step / 6000)
        scheduler_decay = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[decay])
        scheduler_decay = {'scheduler': scheduler_decay,
                           'interval': 'step',  # or 'epoch'
                           'frequency': 1}

        warm_up = lambda step: 0.0001 / (0.0001 ** (step / 2000)) if (step < 2000) else 1
        scheduler_warm_up = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[warm_up])
        scheduler_warm_up = {'scheduler': scheduler_warm_up,
                             'interval': 'step',  # or 'epoch'
                             'frequency': 1}

        return [optimizer], [scheduler_warm_up, scheduler_decay]

    def training_step(self, batch, batch_idx):
        """
        This method is specific to pytorch lightning.
        It is called for each minibatch that the model should be trained for.
        Basically a part of the normal training loop is just moved here.

        model.train() is already set!
        :param batch: (data, target) of batch size
        :param batch_idx: the id of this batch e.g. for discounting?
        :return:
        """
        loss = []

        # (batch_previous, batch_current), batch_targets
        x, _, trans = batch

        # Values for dynamicness threshold
        epes_stat_err = torch.tensor([], device=self.device)
        epes_dyn_err = torch.tensor([], device=self.device)
        dynamicness = torch.tensor([], device=self.device)

        # Forward pass of the slim
        predictions_fw, predictions_bw, previous_batch_pc, current_batch_pc = self(x, trans)

        # parsing the data from decoder
        fw_pointwise = predictions_fw[-1][0]
        fw_trans = fw_pointwise["static_aggr_trans_matrix"]

        bw_pointwise = predictions_bw[-1][0]
        bw_trans = bw_pointwise["static_aggr_trans_matrix"]

        # Temporal split here todo if more samples in batch than 1, it needs to be verified
        previous_pcl = (previous_batch_pc[..., :3] + previous_batch_pc[..., 3:6])  # from pillared to original pts
        current_pcl = (current_batch_pc[..., :3] + current_batch_pc[..., 3:6])  # from pillared to original pts

        """
        ╔══════════════════════════════════╗
        ║     COMPUTING A FORWARD LOSS     ║
        ╚══════════════════════════════════╝
        """

        fw_raw_flow = fw_pointwise['dynamic_flow']  # flow from raft (raw flow)
        fw_rigid_flow = fw_pointwise['static_aggr_flow']  # flow from kabsch

        # nearest neighbor for dynamic flow
        fw_raw_flow_nn_error, fw_raw_flow_nn_idx = NN_loss(previous_pcl + fw_raw_flow, current_pcl, reduction='none')

        # nearest neighbor for rigid (kabsch) flow
        fw_rigid_flow_nn_error, _ = NN_loss(previous_pcl + fw_rigid_flow, current_pcl, reduction='none')

        # Update values for dynamicness thresholds
        epes_stat_err = torch.cat((epes_stat_err, fw_rigid_flow_nn_error[0]))
        epes_dyn_err = torch.cat((epes_dyn_err, fw_raw_flow_nn_error[0]))
        dynamicness = torch.cat((dynamicness, fw_pointwise["dynamicness"][0, :, 0]))

        # Artificial loss
        is_static_artificial_label = fw_rigid_flow_nn_error < fw_raw_flow_nn_error
        art_loss = self._loss_BCE(input=fw_pointwise["staticness"][:, :, 0], target=is_static_artificial_label.float())

        # Smoothness loss
        smooth_loss = smoothness_loss(p_i=previous_pcl, est_flow=fw_raw_flow, K=5, reduction='mean')

        # Nearest neighbour loss for forward
        nn_loss = (fw_raw_flow_nn_error + fw_rigid_flow_nn_error).mean()

        # todo remove outlier percentage - Does not needed

        # Static Points Loss # TODO i added detach on staticness learn it only from artificial loss
        stat_loss = static_point_loss(previous_pcl, T=fw_trans, static_flow=fw_pointwise["static_flow"],
                                      staticness=fw_pointwise["staticness"].detach(), reduction="mean")

        # Rigid Cycle
        cycle_loss = rigid_cycle_loss(previous_pcl, fw_trans, bw_trans)

        # Metric calculation
        # valid_flow_mask = torch.ones_like(y[..., 3], dtype=torch.bool, device=self.device)  # todo add
        # epe, acc_strict, acc_relax = eval_flow(y[..., :3], fw_raw_flow_i)

        loss.append(2. * nn_loss + 1. * cycle_loss + 1 * stat_loss + 0.1 * art_loss + 1 * smooth_loss)

        """
        ╔══════════════════════════════════╗
        ║     COMPUTING A BACKWARD LOSS    ║
        ╚══════════════════════════════════╝
        """

        bw_raw_flow = bw_pointwise['dynamic_flow']
        bw_rigid_flow = bw_pointwise['static_aggr_flow']

        bw_raw_flow_nn_error, _ = NN_loss(current_pcl + bw_raw_flow, previous_pcl, reduction='none')
        bw_rigid_flow_nn_error, _ = NN_loss(current_pcl + bw_rigid_flow, previous_pcl, reduction='none')

        # Update values for dynamicness thresholds
        epes_stat_err = torch.cat((epes_stat_err, bw_rigid_flow_nn_error[0]))
        epes_dyn_err = torch.cat((epes_dyn_err, bw_raw_flow_nn_error[0]))
        dynamicness = torch.cat((dynamicness, bw_pointwise["dynamicness"][0, :, 0]))

        # Artificial loss
        is_static_artificial_label = bw_rigid_flow_nn_error < bw_raw_flow_nn_error
        bw_art_loss = self._loss_BCE(input=bw_pointwise["staticness"][:, :, 0],
                                     target=is_static_artificial_label.float())

        # Smoothness loss
        bw_smooth_loss = smoothness_loss(p_i=current_pcl, est_flow=bw_raw_flow, K=5, reduction='mean')

        # NN Loss
        bw_nn_loss = (bw_raw_flow_nn_error + bw_rigid_flow_nn_error).mean()

        # Static Points Loss
        bw_stat_loss = static_point_loss(current_pcl, T=bw_trans, static_flow=bw_pointwise["static_flow"],
                                         staticness=bw_pointwise["staticness"].detach(), reduction="mean")

        # Rigid Cycle
        bw_cycle_loss = rigid_cycle_loss(current_pcl, bw_trans, fw_trans)

        loss.append(
            2. * bw_nn_loss + 1. * bw_cycle_loss + 1 * bw_stat_loss + 0.1 * bw_art_loss + 1 * bw_smooth_loss)

        loss = loss[0] + loss[1]

        # Update moving average threshold
        self._moving_dynamicness_threshold.update(
            epes_stat_flow=epes_stat_err,
            epes_dyn_flow=epes_dyn_err,
            dynamicness=dynamicness,
            training=True)

        # Log all loss
        phase = "train"
        self.log(f'{phase}/fw/loss/nn', nn_loss.item(), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'{phase}/fw/loss/rigid_cycle', cycle_loss.item(), on_step=True, on_epoch=True, prog_bar=False,
                 logger=True)
        self.log(f'{phase}/fw/loss/artificial', art_loss.item(), on_step=True, on_epoch=True, prog_bar=False,
                 logger=True)
        self.log(f'{phase}/fw/loss/static_point_loss', stat_loss.item(), on_step=True, on_epoch=True,
                 prog_bar=False, logger=True)
        self.log(f'{phase}/fw/loss/smoothness', smooth_loss.item(), on_step=True, on_epoch=True, prog_bar=False,
                 logger=True)

        self.log(f"{phase}/lr", self.optimizers().optimizer.param_groups[0]["lr"], on_step=True, logger=True)
        self.log(f'{phase}/moving_threshold', self._moving_dynamicness_threshold.value(), on_step=True, on_epoch=True,
                 prog_bar=False, logger=True)
        self.log(f'{phase}/loss/', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log(f'{phase}/bw/loss/nn', bw_nn_loss.item(), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log(f'{phase}/bw/loss/rigid_cycle', bw_cycle_loss.item(), on_step=True, on_epoch=True, prog_bar=False,
                 logger=True)
        self.log(f'{phase}/bw/loss/artificial', bw_art_loss.item(), on_step=True, on_epoch=True, prog_bar=False,
                 logger=True)
        self.log(f'{phase}/bw/loss/static_point_loss', bw_stat_loss.item(), on_step=True, on_epoch=True,
                 prog_bar=False, logger=True)
        self.log(f'{phase}/bw/loss/smoothness', bw_smooth_loss.item(), on_step=True, on_epoch=True, prog_bar=False,
                 logger=True)

        self.log(f'{phase}/fw/staticness', fw_pointwise["staticness"].mean().detach().cpu(), on_step=True,
                 on_epoch=True,
                 prog_bar=False, logger=True)
        self.log(f'{phase}/fw/dynamicness', fw_pointwise["dynamicness"].mean().detach().cpu(), on_step=True,
                 on_epoch=True, prog_bar=False, logger=True)

        self.last_output = [previous_batch_pc.detach(), current_batch_pc.detach(), trans.detach(), fw_pointwise]

        # Return loss for backpropagation
        return loss

    def test_step(self, batch, batch_idx):
        """
        Similar to the train step.
        Already has model.eval() and torch.nograd() set!
        :param batch:
        :param batch_idx:
        :return:
        """
        # (batch_previous, batch_current), batch_targets
        x, gt_flow, trans = batch

        # Forward pass of the slim
        predictions_fw, predictions_bw, previous_batch_pc, current_batch_pc = self(x, trans)

        # parsing the data from decoder
        fw_pointwise = predictions_fw[-1][0]
        flow = fw_pointwise["aggr_flow"]

        err = torch.linalg.vector_norm((gt_flow - flow), ord=2, dim=0).mean()


if __name__ == "__main__":
    ### CONFIG ###
    cfg = load_config("../configs/slim.yaml")

    ### DATAMODULE ###
    from datasets.waymoflow.waymodatamodule import WaymoDataModule
    from datasets.kitti.kittidatamodule import KittiDataModule

    DATASET = "waymo"
    grid_cell_size = 0.109375
    data_cfg = cfg["data"][DATASET]

    ### MODEL ####
    model = SLIM(config=cfg, dataset=DATASET)

    if DATASET == "waymo":
        dataset_path = "../data/waymoflow_subset"
        # dataset_path = "/Users/simonpokorny/mnt/data/waymo/raw/processed/training"
        data_cfg["point_features"] = 8
        data_module = WaymoDataModule(dataset_directory=dataset_path,
                                      grid_cell_size=grid_cell_size,
                                      x_min=-35,
                                      x_max=35,
                                      y_min=-35,
                                      y_max=35,
                                      z_min=data_cfg["z_min"],
                                      z_max=10,
                                      batch_size=1,
                                      has_test=False,
                                      num_workers=0,
                                      n_pillars_x=640,
                                      n_points=None, apply_pillarization=True)
    elif DATASET == "rawkitti":
        dataset_path = "../data/rawkitti"
        # dataset_path = "/Users/simonpokorny/mnt/data/waymo/raw/processed/training"
        data_module = KittiDataModule(dataset_directory=dataset_path,
                                      grid_cell_size=grid_cell_size,
                                      x_min=-35,
                                      x_max=35,
                                      y_min=-35,
                                      y_max=35,
                                      z_min=data_cfg["z_min"],
                                      z_max=10,
                                      batch_size=1,
                                      has_test=False,
                                      num_workers=0,
                                      n_pillars_x=640,
                                      n_points=None, apply_pillarization=True)
    else:
        raise ValueError()

    ### TRAINER ###
    loggers = TensorBoardLogger(save_dir="", log_graph=True, version=0)

    trainer = pl.Trainer(fast_dev_run=True, num_sanity_val_steps=0)  # Add Trainer hparams if desired
    trainer.fit(model, data_module)
    # trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    print("done")
