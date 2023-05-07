import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from losses.flow import rigid_cycle_loss, NN_loss, smoothness_loss, static_point_loss
from metrics import AccS, AccR, AEE, AEE_50_50, Outl, ROutl
from models.networks import PillarFeatureNetScatter, PointFeatureNet, MovingAverageThreshold, RAFT
from models.networks.slimdecoder import OutputDecoder
from models.utils import init_weights, get_pointwise_pillar_coords, create_bev_occupancy_grid


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

        self.last_output = None
        self._loss_BCE = torch.nn.BCELoss()

        self.save_hyperparameters()  # Store the constructor parameters into self.hparams
        self.n_features = config["data"][dataset]["point_features"] + 3
        self.n_pillars_x = config["data"][dataset]["n_pillars_x"]
        self.n_pillars_y = config["data"][dataset]["n_pillars_y"]

        self._point_feature_net = PointFeatureNet(in_features=self.n_features, out_features=64)
        self._point_feature_net.apply(init_weights)

        self._pillar_feature_net = PillarFeatureNetScatter(n_pillars_x=self.n_pillars_x, n_pillars_y=self.n_pillars_y)
        self._pillar_feature_net.apply(init_weights)

        # Raft init weights is done internally.
        self._raft = RAFT(**config["raft"])

        self._moving_dynamicness_threshold = MovingAverageThreshold(**config["moving_threshold"][dataset])

        self._decoder_fw = OutputDecoder(**config["decoder"])
        self._decoder_bw = OutputDecoder(**config["decoder"])

        # Metrics for loging the performance of the network
        self.accr = AccR()
        self.accs = AccS()
        self.outl = Outl()
        self.routl = ROutl()
        self.aee = AEE()
        self.aee_50_50 = AEE_50_50(scanning_frequency=10)

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

    def forward(self, x, transforms_matrice):
        """
        The usual forward pass function of a torch module
        :param x: A tuple (previous_pcl, current_pcl) where each batch of point cloud has the same structure.
            previous_pcl[0] - Tensor of point clouds after pillarization in shape (BS, num points, num features)
            previous_pcl[1] - Grid indices in flattened mode in shape (BS, num points)
            previous_pcl[2] - Boolean mask tensor in shape (BS, num points) indicating if the point is valid.
        :param transforms_matrice: A tensor for the transformation matrix from previous to current point cloud in shape (BS, 4, 4).
        :return:
            - predictions_fw: Forward predictions
            - predictions_bw: Backward predictions
            - previous_batch_pc: Tensor of previous point cloud batch in shape (BS, num points, num features)
            - current_batch_pc: Tensor of current point cloud batch in shape (BS, num points, num features)

        """
        # 1. Do scene encoding of each point cloud to get the grid with pillar embeddings

        # The input here is more complex as we deal with a batch of point clouds
        # that do not have a fixed amount of points
        # x is a tuple of two lists representing the batches
        previous_batch, current_batch = x

        # trans is a tensor representing transforms from previous to current frame (t0 -> t1)
        P_T_C = transforms_matrice.type(self.dtype)
        C_T_P = torch.linalg.inv(P_T_C).type(self.dtype)

        previous_batch_pc, previous_batch_grid, previous_batch_mask = previous_batch
        current_batch_pc, current_batch_grid, current_batch_mask = current_batch
        assert current_batch_pc.shape[2] == self.n_features and previous_batch_pc.shape[2] == self.n_features

        previous_batch_pc = previous_batch_pc.type(self.dtype)
        current_batch_pc = current_batch_pc.type(self.dtype)

        # Get point-wise voxel coods in shape [BS, num points, 2],
        current_voxel_coordinates = get_pointwise_pillar_coords(current_batch_grid, self.n_pillars_x, self.n_pillars_y)
        previous_voxel_coordinates = get_pointwise_pillar_coords(previous_batch_grid, self.n_pillars_x, self.n_pillars_y)

        # Create bool map of filled/non-filled pillars
        current_batch_pillar_mask = create_bev_occupancy_grid(current_voxel_coordinates, current_batch_mask,
                                                              self.n_pillars_x, self.n_pillars_y)
        previous_batch_pillar_mask = create_bev_occupancy_grid(previous_voxel_coordinates, previous_batch_mask,
                                                              self.n_pillars_x, self.n_pillars_y)

        # Pass the whole batch of point clouds to get the embedding for each point in the cloud
        # Input pc is (batch_size, max_n_points, features_in)
        previous_batch_pc_embedding = self._transform_point_cloud_to_embeddings(previous_batch_pc,
                                                                                previous_batch_mask).type(self.dtype)
        # previous_batch_pc_embedding = [n_batch, N, 64]
        # Output pc is (batch_size, max_n_points, embedding_features)
        current_batch_pc_embedding = self._transform_point_cloud_to_embeddings(current_batch_pc,
                                                                               current_batch_mask).type(self.dtype)

        # Now we need to scatter the points into their 2D matrix
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

        # 2. RAFT Encoder with motion flow backbone
        # Output for forward pass and backward pass
        # Each of the output is a list of num_iters x [1, 9, n_pillars_x, n_pillars_x]
        # logits, static_flow, dynamic_flow, weights are concatinate in channels in shapes [4, 2, 2, 1]
        outputs_fw, outputs_bw = self._raft(pillar_embeddings)

        # 3. SLIM Decoder
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
                odom=P_T_C,
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

        scheduler_lambda = lambda step: 0.0001 / (0.0001 ** (step / 2000)) if (step < 2000) else 0.5 ** int(step / 6000)
        scheduler_warm_up = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[scheduler_lambda])

        scheduler = {'scheduler': scheduler_warm_up,
                     'interval': 'step',  # or 'epoch'
                     'frequency': 1}

        return [optimizer], [scheduler]

    def compute_loss(self, pointwise_output, previous_pcl, current_pcl, inverse_trans, phase="train", mode="fw"):
        # Dynamic flow from RAFT
        raw_flow = pointwise_output['dynamic_flow']
        # Static aggregated flow from Kabsch
        rigid_flow = pointwise_output['static_aggr_flow']

        # nearest neighbor for dynamic flow
        raw_flow_nn_error = NN_loss(previous_pcl + raw_flow, current_pcl, reduction='none')

        # nearest neighbor for rigid (kabsch) flow
        rigid_flow_nn_error = NN_loss(previous_pcl + rigid_flow, current_pcl, reduction='none')

        # Update moving average threshold
        self._moving_dynamicness_threshold.update(
            epes_stat_flow=rigid_flow_nn_error[0].detach(),
            epes_dyn_flow=raw_flow_nn_error[0].detach(),
            dynamicness=pointwise_output["dynamicness"][0, :, 0],
            training=True)

        # Artificial loss
        is_static_artificial_label = rigid_flow_nn_error < raw_flow_nn_error
        art_loss = self._loss_BCE(input=pointwise_output["staticness"][:, :, 0],
                                  target=is_static_artificial_label.float())

        # Smoothness loss
        smooth_loss = smoothness_loss(p_i=previous_pcl, est_flow=raw_flow, K=5, reduction='mean')

        # Nearest neighbour loss for forward
        nn_loss = (raw_flow_nn_error + rigid_flow_nn_error).mean()

        # Static Points Loss
        stat_loss = static_point_loss(previous_pcl,
                                      T=pointwise_output["static_aggr_trans_matrix"],
                                      static_flow=pointwise_output["static_flow"],
                                      staticness=pointwise_output["staticness"].detach(),
                                      reduction="mean")

        # Rigid Cycle
        cycle_loss = rigid_cycle_loss(previous_pcl, pointwise_output["static_aggr_trans_matrix"], inverse_trans)

        self.log_losses(phase=phase, mode=mode,
                        nn=nn_loss,
                        rigid_cycle=cycle_loss,
                        artificial=art_loss,
                        static_point_loss=stat_loss,
                        smoothness=smooth_loss)
        loss = 2. * nn_loss + 1. * cycle_loss + 1 * stat_loss + 0.1 * art_loss + 1 * smooth_loss
        return loss

    def log_losses(self, phase, mode, **kwargs):
        for name, value in kwargs.items():
            self.log(f'{phase}/{mode}/loss/{name}', value.item(), on_step=True, on_epoch=True, logger=True)


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

        # Compute forward and backward loss
        loss.append(self.compute_loss(fw_pointwise, previous_pcl, current_pcl, bw_trans, phase="train", mode="fw"))
        loss.append(self.compute_loss(bw_pointwise, current_pcl, previous_pcl, fw_trans, phase="train", mode="bw"))

        loss = loss[0] + loss[1]

        # Log others
        phase = "train"
        mean_staticness = fw_pointwise["staticness"].mean().detach().cpu()
        mean_dynamicness = fw_pointwise["dynamicness"].mean().detach().cpu()
        threshold = self._moving_dynamicness_threshold.value()
        self.log(f"{phase}/lr", self.optimizers().optimizer.param_groups[0]["lr"], on_step=True, logger=True)
        self.log(f'{phase}/moving_threshold', threshold, on_step=True, on_epoch=True, logger=True)
        self.log(f'{phase}/loss/', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{phase}/fw/staticness', mean_staticness, on_step=True, on_epoch=True, logger=True)
        self.log(f'{phase}/fw/dynamicness', mean_dynamicness, on_step=True, on_epoch=True, logger=True)

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
        flow = fw_pointwise["aggregated_flow"]
        previous_pcl = (previous_batch_pc[..., :3] + previous_batch_pc[..., 3:6])  # from pillared to original pts
        self.last_output = [previous_batch_pc.detach(), current_batch_pc.detach(), trans.detach(), fw_pointwise]

        # Computing all metrics
        self.accr(flow=flow, gt_flow=gt_flow)
        self.accs(flow=flow, gt_flow=gt_flow)
        self.outl(flow=flow, gt_flow=gt_flow)
        self.routl(flow=flow, gt_flow=gt_flow)
        self.aee(flow=flow, gt_flow=gt_flow)

        if (trans[0] == torch.eye(4, device=self.device)).all():
            self.have_odometry = False
        else:
            self.aee_50_50(flow=flow, gt_flow=gt_flow, odometry=trans, pcl_t0=previous_pcl)
            self.have_odometry = True

        phase = "test"
        self.log(f'{phase}/accr', self.accr.compute(), on_step=True, on_epoch=True)
        self.log(f'{phase}/accs', self.accs.compute(), on_step=True, on_epoch=True)
        self.log(f'{phase}/outl', self.outl.compute(), on_step=True, on_epoch=True)
        self.log(f'{phase}/routl', self.routl.compute(), on_step=True, on_epoch=True)
        self.log(f'{phase}/aee', self.aee.compute(), on_step=True, on_epoch=True, prog_bar=True)

        if self.have_odometry:
            avg, stat, dyn = self.aee_50_50.compute()
            self.log(f'{phase}/aee_50_50/average', avg, on_step=True, on_epoch=True)
            self.log(f'{phase}/aee_50_50/static', stat, on_step=True, on_epoch=True)
            self.log(f'{phase}/aee_50_50/dynamic', dyn, on_step=True, on_epoch=True)

            num_stat, num_dyn = self.aee_50_50.compute_total()
            self.log(f'{phase}/aee_50_50/static_percentage', num_stat, on_step=True, on_epoch=True)
            self.log(f'{phase}/aee_50_50/dynamic_percentage', num_dyn, on_step=True, on_epoch=True)


if __name__ == "__main__":
    DATASET = "nuscenes"
    trained_on = "nuscenes"
    assert DATASET in ["waymo", "rawkitti", "kittisf", "nuscenes"]

    from configs import get_config
    from datasets import get_datamodule

    cfg = get_config("../configs/slim.yaml", dataset=DATASET)
    model = SLIM(config=cfg, dataset=trained_on)
    #model = model.load_from_checkpoint("waymo100k.ckpt")

    cfg["data"][DATASET]["num_workers"] = 0
    data_module = get_datamodule(dataset=DATASET, data_path=None, cfg=cfg)

    loggers = TensorBoardLogger(save_dir="", log_graph=True, version=0)
    trainer = pl.Trainer(fast_dev_run=True, num_sanity_val_steps=0, logger=loggers)  # Add Trainer hparams if desired

    trainer.fit(model, data_module)
    trainer.test(model, data_module)
    # trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    print("Done")
