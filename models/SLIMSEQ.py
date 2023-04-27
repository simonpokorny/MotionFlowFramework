from models import SLIM
import torch


class SLIMSEQ(SLIM):
    def __init__(self, config, dataset):
        """
        Args:
            config (dict): Config is based on configs from configs/slim.yaml
            dataset (str): type of the dataset
        """
        super(SLIM, self).__init__(config, dataset)
        self.save_hyperparameters()  # Store the constructor parameters into self.hparams
        assert type(config) == dict

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
        if predictions_bw == None:
            return torch.zeros((1,), device=self.device, requires_grad=True)
        try:
            predictions_fw, predictions_bw, previous_batch_pc, current_batch_pc = self(x, trans)
        except:
            return torch.zeros((1,), device=self.device, requires_grad=True)

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



