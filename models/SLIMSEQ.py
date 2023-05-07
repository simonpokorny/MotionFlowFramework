import sys

sys.path.append('../../')
sys.path.append('../')

from models import SLIM
import torch
from losses.flow import rigid_cycle_loss, smoothness_loss, NN_loss, static_point_loss
from pytorch3d.ops.knn import knn_points
from datasets.base.utils import ApplyPillarization

class SLIMSEQ(SLIM):
    def __init__(self, config, dataset):
        """
        Args:
            config (dict): Config is based on configs from configs/slim.yaml
            dataset (str): type of the dataset
        """
        super(SLIMSEQ, self).__init__(config, dataset)
        self.save_hyperparameters()  # Store the constructor parameters into self.hparams
        assert type(config) == dict

        self.pillarization = ApplyPillarization(grid_cell_size=0.109375, x_min=-35, y_min=-35, z_min=0.45, z_max=10,
                                                n_pillars_x=640)

        # overwrite the contraction of the class and probs
        self._decoder_fw._constract_probs_and_class = self._constract_probs_and_class



    def _constract_probs_and_class(self, network_output_dict, dynamicness_threshold):

        # Concatination of all logits
        network_output_dict["class_logits"] = torch.cat(
            [network_output_dict["static_logit"],
             network_output_dict["dynamic_logit"],
             network_output_dict["ground_logit"]], dim=1)

        # Softmax on class probs
        network_output_dict["class_probs"] = torch.nn.functional.softmax(network_output_dict["class_logits"], dim=1)

        # Probs of invidual classes are separated into individual keys in dict
        network_output_dict["staticness"] = network_output_dict["class_probs"][:, 0:1]
        network_output_dict["dynamicness"] = network_output_dict["class_probs"][:, 1:2]
        network_output_dict["groundness"] = network_output_dict["class_probs"][:, 2:3]

        # Creating the output based on probs of individual classes and dynamicness threshold
        # DYNAMIC - dynamisness > dynamicness_threshold
        classes = torch.argmax(network_output_dict["class_probs"], dim=1)[None, ...]
        is_dynamic = (classes == 1 )
        is_static = (classes == 0)
        is_ground = (classes == 2)
        # Class prediction in one-hot encoding
        network_output_dict["class_prediction"] = torch.cat([is_static, is_dynamic, is_ground], dim=1)
        network_output_dict["is_static"] = is_static
        return network_output_dict



    def training_sub_step(self, predictions_fw, predictions_bw, previous_batch_pc, current_batch_pc):
        loss_act = []
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


        loss_act.append(2. * nn_loss + 1. * cycle_loss + 1 * stat_loss + 0.1 * art_loss + 1 * smooth_loss)

        """
        ╔══════════════════════════════════╗
        ║     COMPUTING A BACKWARD LOSS    ║
        ╚══════════════════════════════════╝
        """

        bw_raw_flow = bw_pointwise['dynamic_flow']
        bw_rigid_flow = bw_pointwise['static_aggr_flow']

        bw_raw_flow_nn_error, _ = NN_loss(current_pcl + bw_raw_flow, previous_pcl, reduction='none')
        bw_rigid_flow_nn_error, _ = NN_loss(current_pcl + bw_rigid_flow, previous_pcl, reduction='none')

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

        loss_act.append(
            2. * bw_nn_loss + 1. * bw_cycle_loss + 1 * bw_stat_loss + 0.1 * bw_art_loss + 1 * bw_smooth_loss)

        loss_act = loss_act[0] + loss_act[1]
        # Log all loss
        return loss_act

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

        pcli , Ti = batch
        coordiantes_grid = [self._batch_grid_2D(x[1]) for x in pcli]
        # (t3_to_t0, t3_to_t1, t3_to_t2) = Ti
        # (t0_frame, t1_frame, t2_frame, t3_frame) = pcli
        loss = []
        forward_flow = []
        dynamicness = []
        loss = 0

        for idx in range(len(pcli)-1):
            x = (pcli[idx], pcli[idx+1])
            trans = Ti[idx]

            # Forward pass of the slim
            predictions_fw, predictions_bw, previous_batch_pc, current_batch_pc = self(x, trans)
            loss_act = self.training_sub_step(predictions_fw, predictions_bw, previous_batch_pc, current_batch_pc)
            loss += loss_act
            forward_flow.append(predictions_fw[-1][0]["aggregated_flow"])
            dynamicness.append(predictions_fw[-1][0]["dynamicness"])

        pcl3 = pcli[3][0][:, :, :3] + pcli[3][0][:, :, 3:6]
        pcl2 = pcli[2][0][:, :, :3] + pcli[2][0][:, :, 3:6]
        pcl1 = pcli[1][0][:, :, :3] + pcli[1][0][:, :, 3:6]
        pcl0 = pcli[0][0][:, :, :3] + pcli[0][0][:, :, 3:6]

        lengths0 = torch.tensor([pcl0.shape[1]], dtype=torch.long, device=self.device)
        lengths1 = torch.tensor([pcl1.shape[1]], dtype=torch.long, device=self.device)
        lengths2 = torch.tensor([pcl2.shape[1]], dtype=torch.long, device=self.device)
        lengths3 = torch.tensor([pcl3.shape[1]], dtype=torch.long, device=self.device)

        idx_2_to_1 = knn_points(pcl2.float(), (pcl1 + forward_flow[1]).float(), lengths1=lengths2, lengths2=lengths1, K=1, norm=1).idx
        choosen_pcl1_by_idx = torch.index_select(pcl1, 1, idx_2_to_1[0, :, 0])
        choosen_flow_1_to_2 = torch.index_select(forward_flow[1], 1, idx_2_to_1[0, :, 0])
        #choosen_pcl1__grid_by_idx = torch.index_select(coordiantes_grid[1], 1, idx_2_to_1[0, :, 0])

        idx_1_to_0 = knn_points(choosen_pcl1_by_idx.float(), (pcl0 + forward_flow[0]).float(), lengths1=lengths1, lengths2=lengths0, K=1, norm=1).idx
        choosen_pcl0_by_idx = torch.index_select(pcl0, 1, idx_1_to_0[0, :, 0])
        choosen_flow_0_to_1 = torch.index_select(forward_flow[0], 1, idx_1_to_0[0, :, 0])
        #choosen_pcl0__grid_by_idx = torch.index_select(coordiantes_grid[0], 1, idx_1_to_0[0, :, 0])

        idx_2_to_3 = knn_points((pcl2+forward_flow[2]).float(), pcl3.float(), lengths1=lengths2, lengths2=lengths3, K=1, norm=1).idx
        choosen_pcl3_by_idx = torch.index_select(pcl3, 1, idx_2_to_3[0, :, 0])
        #choosen_pcl3__grid_by_idx = torch.index_select(coordiantes_grid[3], 1, idx_2_to_3[0, :, 0])


        import open3d as o3d
        import numpy as np

        NUM_POINTS = 1000

        flow0 = choosen_flow_0_to_1[0].detach().numpy()[:NUM_POINTS]
        flow1 = choosen_flow_1_to_2[0].detach().numpy()[:NUM_POINTS]
        flow2 = forward_flow[2][0].detach().numpy()[:NUM_POINTS]

        pcl0 = choosen_pcl0_by_idx[0].detach().numpy()[:NUM_POINTS]
        pcl1 = choosen_pcl1_by_idx[0].detach().numpy()[:NUM_POINTS]
        pcl2_ = pcl2[0].detach().numpy()[:NUM_POINTS]

        indices = np.arange(NUM_POINTS * 2).reshape((2, -1)).T

        from datasets.visualization.utils import create_o3d_pcl

        t0_frame_o3d = create_o3d_pcl(pcl0, [0.1, 0.6, 0.1])
        t1_frame_o3d = create_o3d_pcl(pcl1, [0.1, 0.1, 0.7])
        t2_frame_o3d = create_o3d_pcl(pcl2_, [0.9, 0.1, 0.1])

        flow0 = np.concatenate([pcl0, flow0 + pcl0], axis=0)
        flow1 = np.concatenate([pcl1, flow1 + pcl1], axis=0)
        flow2 = np.concatenate([pcl2_, flow2 + pcl2_], axis=0)

        o3d_flow0 = o3d.geometry.LineSet()
        o3d_flow0.points = o3d.utility.Vector3dVector(flow0)
        o3d_flow0.lines = o3d.utility.Vector2iVector(indices)

        o3d_flow1 = o3d.geometry.LineSet()
        o3d_flow1.points = o3d.utility.Vector3dVector(flow1)
        o3d_flow1.lines = o3d.utility.Vector2iVector(indices)

        o3d_flow2 = o3d.geometry.LineSet()
        o3d_flow2.points = o3d.utility.Vector3dVector(flow2)
        o3d_flow2.lines = o3d.utility.Vector2iVector(indices)

        o3d.visualization.draw_geometries([t0_frame_o3d, t1_frame_o3d, t2_frame_o3d, o3d_flow0,o3d_flow1, o3d_flow2])




        ### VISIBILITY ###

        class_target = torch.zeros_like(pcl2, device=self.device, dtype=torch.bool)


        class_target[0, :, 0:1] = self.check_visibility(choosen_pcl0_by_idx + choosen_flow_0_to_1, choosen_pcl1_by_idx)
        class_target[0, :, 1:2] = self.check_visibility(choosen_pcl1_by_idx + choosen_flow_1_to_2, pcl2)
        class_target[0, :, 2:3] = self.check_visibility(pcl2 + forward_flow[2], choosen_pcl3_by_idx)



        # torch.unique(class_target.sum(2), return_counts=True)

        mask_dyn = class_target.sum(2) == 3
        # from visualization.plot import save_pcl_class
        # save_pcl_class(pcl2, mask_dyn, ".", "name", ["g", "b"], ["stat", "dynamic"], show=True)


        vis_loss = self._loss_BCE(input=dynamicness[2][mask_dyn], target=torch.ones((mask_dyn.sum(), 1), device=self.device))
        dynamicness1 = torch.index_select(dynamicness[1], 1, idx_2_to_1[0, :, 0])[mask_dyn]
        vis_loss += self._loss_BCE(dynamicness1, target=torch.ones((mask_dyn.sum(), 1), device=self.device))
        dynamicness0 = torch.index_select(dynamicness[0], 1, idx_1_to_0[0, :, 0])[mask_dyn]
        vis_loss += self._loss_BCE(dynamicness0, target=torch.ones((mask_dyn.sum(), 1), device=self.device))

        ## NORM SMOOTHNESS ##

        f0_norm = torch.linalg.norm(choosen_flow_0_to_1, dim=2)
        f1_norm = torch.linalg.norm(choosen_flow_1_to_2, dim=2)
        f2_norm = torch.linalg.norm(forward_flow[2], dim=2)

        diff0 = f0_norm - f1_norm
        diff1 = f1_norm - f2_norm

        loss_speed = torch.nn.functional.mse_loss(input=diff0, target=diff1, reduction='mean')
        #loss_speed += torch.nn.functional.mse_loss(input=diff1, target=diff0, reduction='mean')


        f0_angle = torch.atan2(choosen_flow_0_to_1[:,:,1], choosen_flow_0_to_1[:,:,0])
        f1_angle = torch.atan2(choosen_flow_1_to_2[:,:,1], choosen_flow_1_to_2[:,:,0])
        f2_angle = torch.atan2(forward_flow[2][:,:,1], forward_flow[2][:,:,0])

        diff_ang0 = f0_angle - f1_angle
        diff_ang1 = f1_angle - f2_angle

        diff_ang0 = torch.where(diff_ang0 > torch.pi, diff_ang0 - 2 * torch.pi, diff_ang0)
        diff_ang0 = torch.where(diff_ang0 <= -torch.pi, diff_ang0 + 2 * torch.pi, diff_ang0)

        diff_ang1 = torch.where(diff_ang1 > torch.pi, diff_ang1 - 2 * torch.pi, diff_ang1)
        diff_ang1 = torch.where(diff_ang1 <= -torch.pi, diff_ang1 + 2 * torch.pi, diff_ang1)

        loss_angle = torch.nn.functional.mse_loss(input=diff_ang0, target=diff_ang1, reduction='mean')
        #loss_angle += torch.nn.functional.mse_loss(input=diff_ang1, target=diff_ang0, reduction='mean')


        phase = "train"
        self.log(f'{phase}/fw/loss/slim_loss', loss.item(), on_step=True, on_epoch=False)
        self.log(f'{phase}/fw/loss/angle_loss', loss_angle.item(), on_step=True, on_epoch=False)
        self.log(f'{phase}/fw/loss/norm_loss', loss_speed.item(), on_step=True, on_epoch=False)
        self.log(f'{phase}/fw/loss/visibility_loss', vis_loss.item(), on_step=True, on_epoch=False)
        self.log(f"{phase}/lr", self.optimizers().optimizer.param_groups[0]["lr"], on_step=True, logger=True)
        self.log(f"{phase}/num_dyn_targets", mask_dyn.sum(), on_step=True, logger=True)


        loss = loss / 3 + 0.1 * vis_loss + loss_angle + loss_speed
        # self.last_output = [previous_batch_pc.detach(), current_batch_pc.detach(), trans.detach(), fw_pointwise]
        # Return loss for backpropagation
        return loss

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
        self.lr = 0.00001
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)



        return [optimizer]

    def visibility_mask(self, pcl_batch_0, pcl_batch_1):
        voxel_coordinates = self._batch_grid_2D(pcl_batch_0[1])
        batch_pillar_mask0 = self._filled_pillar_mask(voxel_coordinates, pcl_batch_0[2])

        voxel_coordinates = self._batch_grid_2D(pcl_batch_1[1])
        batch_pillar_mask1 = self._filled_pillar_mask(voxel_coordinates, pcl_batch_1[2])

        return batch_pillar_mask0 - batch_pillar_mask1

    def check_visibility(self, pcl0, pcl1):

        _, grid0 = self.pillarization(pcl0[0].detach().cpu().numpy())

        _, grid1 = self.pillarization(pcl1[0].detach().cpu().numpy())

        grid0 = torch.tensor(grid0, device=self.device)
        grid1 = torch.tensor(grid1, device=self.device)

        voxel_coordinates0 = self._batch_grid_2D(grid0)
        voxel_coordinates1 = self._batch_grid_2D(grid1)

        pillar_mask0 = torch.zeros((1, 1, 640, 640), device=self.device)
        pillar_mask1 = torch.zeros((1, 1, 640, 640), device=self.device)

        x0 = voxel_coordinates0[..., 0]
        y0 = voxel_coordinates0[..., 1]

        mask = ((x0 >= 0) & (x0 < 640)) & ((y0 < 640) & (y0 >=0))

        x0 = x0[mask]
        y0 = y0[mask]
        grid0 = torch.where(mask, grid0, torch.tensor(0, device=self.device))
        grid0= torch.where(grid0>=0, grid0, torch.tensor(0, device=self.device))


        pillar_mask0[:, :, x0, y0] = torch.tensor(1., device=self.device)

        x1 = voxel_coordinates1[..., 0]
        y1 = voxel_coordinates1[..., 1]

        mask = ((x1 >= 0) & (x1 < 640)) & ((y1 < 640) & (y1 >= 0))

        x1 = x1[mask]
        y1 = y1[mask]
        grid1 = torch.where(mask, grid1, torch.tensor(0, device=self.device))
        grid1= torch.where(grid1>=0, grid1, torch.tensor(0, device=self.device))


        pillar_mask1[:, :, x1, y1] = torch.tensor(1., device=self.device)

        sub_mask = pillar_mask0 - pillar_mask1

        flattened_grid = sub_mask.view(1, -1)

        print(f"_{grid0.min()}_{grid0.max()}_".center(100, "#"))
        gathered_values0 = torch.gather(flattened_grid, 1, grid0[None,...])
        gathered_values1 = torch.gather(flattened_grid, 1, grid1[None,...])


        return (torch.abs(gathered_values1) + torch.abs(gathered_values0)).bool().reshape((1,-1,1))

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
        self.log(f'{phase}/aee', self.aee.compute(), on_step=True, on_epoch=True)

        if self.have_odometry:
            avg, stat, dyn = self.aee_50_50.compute()
            self.log(f'{phase}/aee_50_50/average', avg, on_step=True, on_epoch=True)
            self.log(f'{phase}/aee_50_50/static', stat, on_step=True, on_epoch=True)
            self.log(f'{phase}/aee_50_50/dynamic', dyn, on_step=True, on_epoch=True)

            num_stat, num_dyn = self.aee_50_50.compute_total()
            self.log(f'{phase}/aee_50_50/static_percentage', num_stat, on_step=True, on_epoch=True)
            self.log(f'{phase}/aee_50_50/dynamic_percentage', num_dyn, on_step=True, on_epoch=True)



if __name__ == "__main__":


    DEBUG = True

    if torch.cuda.is_available():
        DEBUG = False

    import os
    import sys
    from pathlib import Path

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

    sys.path.append('../../')
    sys.path.append('../')

    from configs import get_config
    from models.SLIM import SLIM
    import warnings
    warnings.filterwarnings("ignore")
    dataset = "waymo"
    from pathlib import Path
    import os
    from configs import get_config


    EXPERIMENT_PATH = Path("experiments")
    os.makedirs(EXPERIMENT_PATH, exist_ok=True)

    # Loading config
    cfg = get_config("../configs/slim.yaml", dataset=dataset)

    if DEBUG:
        cfg["data"][dataset]["num_workers"] = 0


    # Creating the model
    model = SLIMSEQ(config=cfg, dataset=dataset)
    # model = model.load_from_checkpoint("/home/pokorsi1/motion_learning/scripts/slim/experiments/nuscenes/checkpoints/version_1/epoch=0-step=8000.ckpt")
    # model = model.load_from_checkpoint("/home/pokorsi1/motion_learning/scripts/slim/experiments/waymo/checkpoints/version_3/epoch=0-step=80000.ckpt")
    model = model.load_from_checkpoint("waymo100k.ckpt")

    # Get datamodule
    data_cfg = cfg["data"][dataset]
    grid_cell_size = (data_cfg["x_max"] + abs(data_cfg["x_min"])) / data_cfg["n_pillars_x"]

    dataset_path = "/home/pokorsi1/data/waymo_flow/preprocess" if not DEBUG else "../data/waymoflow"
    data_cfg["has_test"] = False
    from datasets.waymo_sequential.waymoseqdatamodule import WaymoSeqDataModule
    data_module = WaymoSeqDataModule(dataset_directory=dataset_path, grid_cell_size=grid_cell_size, **data_cfg)

    try:
        version = len(os.listdir(os.path.join(EXPERIMENT_PATH, "waymo_seq", "lightning_logs")))
    except:
        version = 0

    print(f"Saved under version num : {version}")

    callbacks = [ModelCheckpoint(dirpath=EXPERIMENT_PATH / "waymo_seq" / "checkpoints" / f"version_{version}",
                                 save_weights_only=False, every_n_train_steps=1000, save_last=True, save_top_k=-1)]

    # SaveViz(dirpath=EXPERIMENT_PATH / args.dataset / "visualization" / f"version_{version}",
    #         every_n_train_steps=1000)]

    loggers = [TensorBoardLogger(save_dir=EXPERIMENT_PATH, name=f"waymo_seq/lightning_logs",
                                 log_graph=True, version=version),
               CSVLogger(save_dir=EXPERIMENT_PATH, name=f"waymo_seq/lightning_logs", version=version)]

    if DEBUG:
        device= "cpu"
    else:
        device = "gpu"

    # trainer with no validation loop
    trainer = pl.Trainer(limit_val_batches=0, num_sanity_val_steps=0, devices=1, accelerator=device,
                         enable_checkpointing=True, max_epochs=10,
                         logger=loggers, callbacks=callbacks)

    trainer.fit(model, data_module)
