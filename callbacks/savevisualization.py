import os
from pathlib import Path

import torch
from pytorch_lightning.callbacks import Callback

from visualization.plot import save_trans_pcl, save_pcl_class, save_pcl_flow


class SaveViz(Callback):
    def __init__(self, dirpath: str, every_n_train_steps: int = 100, every_n_test_steps: int = 1):
        super(SaveViz, self).__init__()
        self.every_n_train_steps = every_n_train_steps
        self.every_n_test_steps = every_n_test_steps
        self.show = False

        self.path = Path(dirpath)
        os.makedirs(self.path, exist_ok=False)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.every_n_train_steps == 0 and batch_idx != 0:
            self.viz_path = self.path / f"batch_idx_{batch_idx}"
            os.makedirs(self.viz_path)

            if pl_module.last_output is not None:

                previous_pcl, current_pcl, P_T_C, fw_pointwise = pl_module.last_output

                fw_trans = fw_pointwise["static_aggr_trans_matrix"]
                save_trans_pcl(fw_trans, P=previous_pcl, C=current_pcl, path=self.viz_path,
                               name="pcl_synchronized_Kabsch", show=self.show)

                probs = fw_pointwise["class_probs"].argmax(2)
                save_pcl_class(tensor=previous_pcl[:, :, :2],
                               classes=probs,
                               path=self.viz_path,
                               name="pcl_class_prediction",
                               colors=["r", "g", "b"],
                               labels=["static", "dynamic", "ground"])

                num_points_to_visualize = 5000
                save_pcl_flow(pcl0=previous_pcl, pcl1=current_pcl, flow=fw_pointwise["dynamic_flow"],
                              odom=P_T_C, path=self.viz_path, name=f"flow_dynamic", show=self.show,
                              num_points=num_points_to_visualize)
                save_pcl_flow(pcl0=previous_pcl, pcl1=current_pcl, flow=fw_pointwise["static_flow"],
                              odom=P_T_C, path=self.viz_path, name=f"flow_static", show=self.show,
                              num_points=num_points_to_visualize)
                save_pcl_flow(pcl0=previous_pcl, pcl1=current_pcl, flow=fw_pointwise["static_aggr_flow"],
                              odom=P_T_C, path=self.viz_path, name=f"flow_static_aggregation", show=self.show,
                              num_points=num_points_to_visualize)
                save_pcl_flow(pcl0=previous_pcl, pcl1=current_pcl, flow=fw_pointwise["aggregated_flow"],
                              odom=P_T_C, path=self.viz_path, name=f"flow_aggregated", show=self.show,
                              num_points=num_points_to_visualize)

                save_trans_pcl(P_T_C=P_T_C, P=previous_pcl, C=current_pcl, path=self.viz_path,
                               name="pcl_synchronized", show=self.show)
                save_trans_pcl(P_T_C=torch.eye(4)[None, :], P=previous_pcl, C=current_pcl, path=self.viz_path,
                               name="pcl_non_synchronized", show=self.show)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx % self.every_n_test_steps == 0:
            self.viz_path = self.path / f"eval_batch_idx_{batch_idx}"
            os.makedirs(self.viz_path)

            if pl_module.last_output is not None:

                previous_pcl, current_pcl, P_T_C, fw_pointwise = pl_module.last_output

                fw_trans = fw_pointwise["static_aggr_trans_matrix"]
                save_trans_pcl(fw_trans, P=previous_pcl, C=current_pcl, path=self.viz_path,
                               name="pcl_synchronized_Kabsch", show=self.show)

                probs = fw_pointwise["class_probs"].argmax(2)
                save_pcl_class(tensor=previous_pcl[:, :, :2],
                               classes=probs,
                               path=self.viz_path,
                               name="pcl_class_prediction",
                               colors=["r", "g", "b"],
                               labels=["static", "dynamic", "ground"])

                num_points_to_visualize = 10000
                save_pcl_flow(pcl0=previous_pcl, pcl1=current_pcl, flow=fw_pointwise["dynamic_flow"],
                              odom=P_T_C, path=self.viz_path, name=f"flow_dynamic", show=self.show,
                              num_points=num_points_to_visualize)
                save_pcl_flow(pcl0=previous_pcl, pcl1=current_pcl, flow=fw_pointwise["static_flow"],
                              odom=P_T_C, path=self.viz_path, name=f"flow_static", show=self.show,
                              num_points=num_points_to_visualize)
                save_pcl_flow(pcl0=previous_pcl, pcl1=current_pcl, flow=fw_pointwise["static_aggr_flow"],
                              odom=P_T_C, path=self.viz_path, name=f"flow_static_aggregation", show=self.show,
                              num_points=num_points_to_visualize)
                save_pcl_flow(pcl0=previous_pcl, pcl1=current_pcl, flow=fw_pointwise["aggregated_flow"],
                              odom=P_T_C, path=self.viz_path, name=f"flow_aggregated", show=self.show,
                              num_points=num_points_to_visualize)

                save_trans_pcl(P_T_C=P_T_C, P=previous_pcl, C=current_pcl, path=self.viz_path,
                               name="pcl_synchronized", show=self.show)
                save_trans_pcl(P_T_C=torch.eye(4)[None, :], P=previous_pcl, C=current_pcl, path=self.viz_path,
                               name="pcl_non_synchronized", show=self.show)
