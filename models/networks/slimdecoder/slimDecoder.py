import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor

from .staticAggregation import StaticAggregatedFlow
from .utils import normalized_sigmoid_sum


class OutputDecoder(pl.LightningModule):
    def __init__(self,
                 predict_weight_for_static_aggregation=True,
                 use_static_aggr_flow_for_aggr_flow=False,
                 use_dynamic_aggr_flow_for_aggr_flow=False,
                 dynamic_flow_is_non_rigid_flow=False,
                 overwrite_non_filled_pillars_with_default_flow: bool = True,
                 overwrite_non_filled_pillars_with_default_logits: bool = True,
                 static_logit="net",
                 dynamic_logit="net",
                 ground_logit=False,
                 use_epsilon_for_weighted_pc_alignment=False,
                 disappearing_logit=False,
                 static_flow="net",
                 dynamic_flow="net",
                 ):
        """
        Initializes the OutputDecoder module.

        Args:
            - predict_weight_for_static_aggregation (bool): If True, predicts weights for static aggregation
            - use_static_aggr_flow_for_aggr_flow (bool): If True, uses static flow for aggregation flow
            - use_dynamic_aggr_flow_for_aggr_flow (bool): If True, uses dynamic flow for aggregation flow
              representing the one-hot encoded ground truth labels for static/dynamic/ground objects in the BEV grid
            - dynamic_flow_is_non_rigid_flow (bool): If True, indicates that the dynamic flow is non-rigid
            - overwrite_non_filled_pillars_with_default_flow (bool): If True, overwrites non-filled pillars with the
                default flow
            - overwrite_non_filled_pillars_with_default_logits (bool): If True, overwrites non-filled pillars with the
                default logits
            - artificial_network_config (dict): Config for ArtificialNetworkOutput()
        """

        super().__init__()

        self.flow_dim = 2

        self.predict_weight_for_static_aggregation = predict_weight_for_static_aggregation
        self.use_static_aggr_flow_for_aggr_flow = use_static_aggr_flow_for_aggr_flow
        self.use_dynamic_aggr_flow_for_aggr_flow = use_dynamic_aggr_flow_for_aggr_flow
        self.dynamic_flow_is_non_rigid_flow = dynamic_flow_is_non_rigid_flow

        self.overwrite_non_filled_pillars_with_default_flow = overwrite_non_filled_pillars_with_default_flow
        self.overwrite_non_filled_pillars_with_default_logits = overwrite_non_filled_pillars_with_default_logits

        self.static_logit = static_logit
        self.dynamic_logit = dynamic_logit
        self.ground_logit = ground_logit
        self.disappearing_logit = disappearing_logit

        self.static_flow = static_flow
        self.dynamic_flow = dynamic_flow

        self.use_epsilon_for_weighted_pc_alignment = use_epsilon_for_weighted_pc_alignment
        self.predict_weight_for_static_aggregation = predict_weight_for_static_aggregation

        self.Kabsch = StaticAggregatedFlow(
            use_eps_for_weighted_pc_alignment=use_epsilon_for_weighted_pc_alignment)

    def forward(self, network_output: Tensor,
                dynamicness_threshold,
                pc: Tensor,
                pointwise_voxel_coordinates_fs: Tensor,
                pointwise_valid_mask,
                filled_pillar_mask,
                odom,
                inv_odom,
                **kwargs):

        """
        Args:
            - network_output: a tensor of shape [batch_size, height, width, num_classes + 2 * num_flow_channels + 1]
              representing the output of a neural network. In our case num of channels should be 9.
                - Disapiring Logit 1
                - Static Logic 1
                - Dynamic Logic 1
                - Ground Logic 1
                - Static flow 2
                - Dynamic flow 2
                - Weights 1
            - dynamicness_threshold: a float value representing the dynamicness threshold used for separating static
              and dynamic objects
            - pc: a tensor of shape [batch_size, num_points, 5 or 4] representing the point cloud input.
              Channels should be in order [x, y, z, feature1, feature2]
            - pointwise_voxel_coordinates_fs: a tensor of shape [batch_size, num_points, 2] representing the coordinates
              of each point in the BEV grid
            - pointwise_valid_mask: a tensor of shape [batch_size, num_points] representing whether each point is valid or not
            - filled_pillar_mask: a tensor of shape [batch_size, height, width] and type bool representing whether each pillar
              in the BEV grid has been filled with points or not
            - inv_odom: a tensor of shape [batch_size, 4, 4] representing the inverse
              of the ground truth transformation matrix

        Returns:
            - pointwise_outputs
            - static_aggr_trafo
            - dynamicness_threshold
            - modified_output_bev_img

        """

        # Check the shapes and dimensions
        assert filled_pillar_mask.ndim == 4
        assert network_output.ndim == 4
        assert filled_pillar_mask.shape[-2:] == network_output.shape[-2:], (
            filled_pillar_mask.shape, network_output.shape)
        assert pointwise_voxel_coordinates_fs.shape[-1] == 2
        # Check the correct dimensions of the output of the network
        assert network_output.shape[1] == 4 + 2 * self.flow_dim + 1
        assert pointwise_voxel_coordinates_fs.shape[-1] == 2

        # Parsing output by channels - CHECKED
        network_output_dict = self._create_output_dict(network_output)
        # if kwargs["it"] == 5 and self.viz and kwargs["mode"] == "fw":
        #    save_tensor_per_channel(network_output, self.viz_path, f"output_decoder",
        #                           labels=["disapiring logit", "static_logit", "dynamic Logit", "ground logit", "STF_x",
        #                                   "STF_y", "DYF_x", "DYN_y", "weights"], show=self.show)

        # Create coordinates of grid (discreate coordinates in real world) - CHECKED (should be writen in torch)
        homog_metric_voxel_center_coords, voxel_center_metric_coordinates = self._create_voxel_coords(network_output)
        # if kwargs["it"] == 5 and self.viz and kwargs["mode"] == "fw":
        #    save_tensor_per_channel(voxel_center_metric_coordinates[None, :].permute(0,3,1,2), self.viz_path,
        #                            labels=["X", "Y"], name="voxels_coords", show=self.show)

        # Create ground truth static flow from odometry - !!! different coord system
        network_output_dict = self._create_gt_static_flow(network_output_dict, inv_odom,
                                                          homog_metric_voxel_center_coords)

        # Compute sources for flows - CHECKED
        network_output_dict = self._source_for_flow(network_output_dict=network_output_dict)

        # Compute sources for logits - CHECKED
        network_output_dict = self._source_for_logits(network_output_dict=network_output_dict)

        # Constructing class probs from logits and decision of classes based on dynamicness threshold - CHECKED
        network_output_dict = self._constract_probs_and_class(network_output_dict=network_output_dict,
                                                              dynamicness_threshold=dynamicness_threshold)

        # Construction of weights for the static aggregation - CHECKED (not sure about normalized sigmoid)
        # If we want use weights logits from network, then static_aggr_weight_map is multiplied
        # by these weights in mode normalized sigmoid or crossentropy
        network_output_dict, masked_static_aggr_weight_map = self._procces_weights_static_aggregation(
            network_output_dict=network_output_dict,
            filled_pillar_mask=filled_pillar_mask)

        # Computing of static aggregation with Kabsch algorithm - CHECKED (not sure for 100%)
        network_output_dict = self._compute_static_aggregated_flow(network_output_dict=network_output_dict,
                                                                   weights=masked_static_aggr_weight_map,
                                                                   pc=pc,
                                                                   pointwise_voxel_coordinates_fs=pointwise_voxel_coordinates_fs,
                                                                   pointwise_valid_mask=pointwise_valid_mask,
                                                                   voxel_center_metric_coordinates=voxel_center_metric_coordinates)

        # if self.viz:
        #    save_trans_pcl(network_output_dict["static_aggr_trans_matrix"], P=pc, C=kwargs["pc_1"], path=self.viz_path,
        #                   name=f"scene_together_from_KABSCH_P_T_C_iter_{kwargs['it']}", show=True)

        # Masking non-filled pillars - CHECKED
        network_output_dict = self._masking_nonfilled_pillars(network_output_dict=network_output_dict,
                                                              filled_pillar_mask=filled_pillar_mask)
        # Making the flow three dimensional - CHECKED
        network_output_dict = self._make_flow_three_dimensional(network_output_dict)

        # Create final flow based on staticness of the points - CHECKED
        network_output_dict["aggregated_flow"] = torch.where(network_output_dict["is_static"],
                                                             network_output_dict["static_aggr_flow"],
                                                             network_output_dict["dynamic_flow"] * (
                                                                         1.0 - network_output_dict["groundness"]))

        # Create BIRD'S EYE VIEW output
        network_output_dict["disappiring"] = torch.sigmoid(network_output_dict["disappearing_logit"])
        modified_output_bev_img = network_output_dict

        # Transform pillars to pointcloud - CHECKED
        pointwise_output = self._apply_flow_to_points(
            network_output_dict=network_output_dict,
            pointwise_voxel_coordinates_fs=pointwise_voxel_coordinates_fs,
            pointwise_valid_mask=pointwise_valid_mask,
        )

        return pointwise_output, dynamicness_threshold, modified_output_bev_img

    def _make_flow_three_dimensional(self, network_output_dict):
        # Which flows make three dimensional
        flow_names = ["dynamic_flow", "static_flow", "static_aggr_flow", "static_gt_flow"]
        for name, flow in network_output_dict.items():
            if name in flow_names:
                network_output_dict[name] = torch.cat([flow, torch.zeros_like(flow[:, :1])], dim=1)
        return network_output_dict

    def _create_output_dict(self, network_output):
        """
        Create the output dict, all decoder use it
        """
        # Partition the output by channels
        network_output_dict = {}
        network_output_dict.update({
            "disappearing_logit": network_output[:, 0:1],
            "static_logit": network_output[:, 1:2],
            "dynamic_logit": network_output[:, 2:3],
            "ground_logit": network_output[:, 3:4],
            "static_flow": network_output[:, 4: 4 + self.flow_dim],
            "dynamic_flow": network_output[:, 4 + self.flow_dim: 4 + 2 * self.flow_dim],
            "weights": network_output[:, 8:]})
        return network_output_dict

    def _compute_static_aggregated_flow(self, network_output_dict, weights, pc, pointwise_voxel_coordinates_fs,
                                        pointwise_valid_mask, voxel_center_metric_coordinates):
        static_aggr_flow, static_aggr_trafo = self.Kabsch(
            static_flow=network_output_dict["static_flow"],
            staticness=weights,
            pc=pc,
            pointwise_voxel_coordinates_fs=pointwise_voxel_coordinates_fs,
            pointwise_valid_mask=pointwise_valid_mask,
            voxel_center_metric_coordinates=voxel_center_metric_coordinates)

        # Adding new values to out dict
        network_output_dict["static_aggr_flow"] = static_aggr_flow
        network_output_dict["static_aggr_trans_matrix"] = static_aggr_trafo
        return network_output_dict

    def _create_gt_static_flow(self, network_output_dict, inv_odom, homog_metric_voxel_center_coords):
        """
        Compute other self supervise component -> gt_static_flow
        """
        # gt_static_flow is (P_T_G - eye), which results in flow, shape
        # TODO should be here inverse?
        # print("UnCheck")
        gt_static_flow = inv_odom - torch.eye(4, device=self.device)
        gt_static_flow = torch.einsum("bij,hwj->bhwi", gt_static_flow.double(),
                                      homog_metric_voxel_center_coords.double()).float()
        # we take only xy coords
        gt_static_flow = gt_static_flow[..., :2]
        # Transform it to default order [BS, CH, H, W]
        gt_static_flow = gt_static_flow.permute(0, 3, 1, 2)
        network_output_dict["static_gt_flow"] = gt_static_flow
        return network_output_dict

    def _source_for_flow(self, network_output_dict):
        """
        Equivalent for ArtificialFlowNetworkOutput in official SLIM implementation

        Computes source from which the flow will be taken.

        Args:
            network_output_dict: A dictionary containing intermediate outputs from the network.

        Returns:
            A dictionary containing the final outputs of the network.
        """

        # We can choose from which source the static flow will be taken.
        # #region static_flow
        if self.static_flow == "net":
            pass
        elif self.static_flow == "gt":
            network_output_dict["static_flow"] = network_output_dict["gt_static_flow"]
        elif self.static_flow == "zero":
            network_output_dict["static_flow"] = torch.zeros_like(network_output_dict["static_flow"])
        else:
            raise ValueError("unknown output mode: %s" % str(self.static_flow))
        # #endregion static_flow

        # We can choose from which source the dynamic flow will be taken.
        # #region dynamic_flow
        if self.dynamic_flow == "net":
            pass
        elif self.dynamic_flow == "gt":
            raise NotImplementedError()
        elif self.dynamic_flow == "zero":
            network_output_dict["dynamic_flow"] = torch.zeros_like(network_output_dict["dynamic_flow"])
        else:
            raise ValueError("unknown output mode: %s" % str(self.dynamic_flow))
        # #endregion dynamic_flow
        return network_output_dict

    def _create_voxel_coords(self, network_output):
        """
        Creation of homog_metric_voxel_center_coords and voxel_center_metric_coordinates
        """
        final_grid_size = network_output.shape[-2:]

        # Creating of bev grid mesh
        bev_extent = np.array([-35.0, -35.0, 35.0, 35.0])
        net_output_shape = final_grid_size  # Net out shape is [640, 640]
        voxel_center_metric_coordinates = (
                np.stack(
                    np.meshgrid(np.arange(net_output_shape[0]), np.arange(net_output_shape[1]), indexing="ij"),
                    axis=-1,
                )
                + 0.5
        )  # now we have voxels in shape [640, 640, 2] from 0.5 to 639.5

        voxel_center_metric_coordinates /= net_output_shape
        voxel_center_metric_coordinates *= bev_extent[2:] - bev_extent[:2]
        voxel_center_metric_coordinates += bev_extent[:2]
        # now we have coordinates with centres of the voxel, for example in
        # voxel_center_metric_coordinates[0, 0] is [-34.9453125, -34.9453125]
        # The resolution (width of voxels) is 70m/640 = 0.1093..
        homog_metric_voxel_center_coords = torch.tensor(np.concatenate(
            [
                voxel_center_metric_coordinates,
                np.zeros_like(voxel_center_metric_coordinates[..., :1]),
                np.ones_like(voxel_center_metric_coordinates[..., :1]),
            ],
            axis=-1,
        ))

        homog_metric_voxel_center_coords = torch.tensor(homog_metric_voxel_center_coords, device=self.device)
        voxel_center_metric_coordinates = torch.tensor(voxel_center_metric_coordinates, device=self.device)

        # homog_metric_voxel_center_coords only add z coord to 0 and 4th dimension for homogeneous coordinates
        return homog_metric_voxel_center_coords, voxel_center_metric_coordinates

    def _apply_flow_to_points(self,
                              network_output_dict,
                              pointwise_voxel_coordinates_fs,
                              pointwise_valid_mask):

        assert (pointwise_voxel_coordinates_fs >= 0).all(), "negative pixel coordinates found"

        bs = pointwise_voxel_coordinates_fs.shape[0]
        x, y = pointwise_voxel_coordinates_fs[:, :, 0], pointwise_voxel_coordinates_fs[:, :, 1]

        pointwise_output = {}
        skip_names = ["static_aggr_trans_matrix"]

        for name, bev_tensor in network_output_dict.items():
            if name in skip_names:
                pointwise_output[name] = bev_tensor
                continue
            pointwise_output[name] = bev_tensor[torch.arange(bs)[:, None], :, x, y]
            pointwise_output[name] = pointwise_output[name][pointwise_valid_mask][None, :]
        return pointwise_output

    def _source_for_logits(self, network_output_dict):
        """
        Equivalent for ArtificialLogitNetwork in SLIM official repo

        We are defining from which source the logits will be taken.

        Parameters:
            network_output_dict (dict): Dictionary containing the output tensors of the network.
            ohe_gt_stat_dyn_ground_label_bev_map (torch.Tensor): Tensor of shape `(B, 3, H, W)` containing the ground-truth
                labels, static flow, and ground truth maps.
            gt_flow_bev (torch.Tensor): Tensor of shape `(B, 2, H, W)` containing the ground-truth flow vectors in the
                BEV space.
            gt_static_flow (torch.Tensor): Tensor of shape `(B, 2, H, W)` containing the ground-truth static flow vectors
                in the BEV space.

        Returns:
            output_dict (dict): Dictionary containing the output tensors of the network with the logits added.
        """

        ones = torch.ones_like(network_output_dict["static_logit"])

        # #region disappearing_logit
        if self.disappearing_logit == "net":
            pass
        elif self.disappearing_logit == "gt":
            raise NotImplementedError()
        elif self.disappearing_logit is True:
            network_output_dict["disappearing_logit"] = 0 * ones
        elif self.disappearing_logit is False:
            network_output_dict["disappearing_logit"] = -100 * ones
        else:
            raise ValueError(
                "unknown output mode: %s" % str(self.disappearing_logit)
            )
        # #endregion disappearing_logit

        # #region static_logit
        if self.static_logit == "net":
            pass
        elif self.static_logit == "gt_label_based":
            raise NotImplementedError()
        elif self.static_logit == "gt_flow_based":
            raise NotImplementedError()

        elif self.static_logit is True:
            assert self.dynamic_logit is False
            assert self.ground_logit is False
            network_output_dict["static_logit"] = (
                    torch.max(torch.stack([network_output_dict["dynamic_logit"],
                                           network_output_dict["ground_logit"]]), dim=0)[0]
                    + 100.0 * ones)
        elif self.static_logit is False:
            assert (self.dynamic_logit is not False or self.ground_logit is not False)
            network_output_dict["static_logit"] = (
                    torch.min(
                        torch.stack([
                            network_output_dict["dynamic_logit"],
                            network_output_dict["ground_logit"]
                        ]),
                        dim=0
                    )[0]
                    - 100.0 * ones
            )
        else:
            raise ValueError("unknown output mode: %s" % str(self.static_logit))
        # #endregion static_logit

        # #region dynamic_logit
        if self.dynamic_logit == "net":
            pass
        elif self.dynamic_logit == "gt_label_based":
            raise NotImplementedError()
        elif self.dynamic_logit == "gt_flow_based":
            raise NotImplementedError()
        elif self.dynamic_logit is True:
            assert self.static_logit is False
            assert self.ground_logit is False
            network_output_dict["dynamic_logit"] = (
                    torch.max(
                        torch.stack([
                            network_output_dict["static_logit"],
                            network_output_dict["ground_logit"]
                        ]),
                        dim=0
                    )[0]
                    + 100.0 * ones
            )
        elif self.dynamic_logit is False:
            network_output_dict["dynamic_logit"] = (
                    torch.min(
                        torch.stack([
                            network_output_dict["static_logit"],
                            network_output_dict["ground_logit"]
                        ]),
                        dim=0
                    )[0]
                    - 100.0 * ones
            )
        else:
            raise ValueError("unknown output mode: %s" % str(self.dynamic_logit))
        # #endregion dynamic_logit

        # #region ground_logit
        if self.ground_logit == "net":
            pass
        elif self.ground_logit == "gt_label_based":
            raise NotImplementedError()
        elif self.ground_logit is True:
            assert self.static_logit is False
            assert self.dynamic_logit is False
            network_output_dict["ground_logit"] = (
                    torch.max(
                        torch.stack([
                            network_output_dict["static_logit"],
                            network_output_dict["dynamic_logit"]
                        ]),
                        dim=0
                    )[0]
                    + 100.0 * ones
            )
        elif self.ground_logit is False:
            network_output_dict["ground_logit"] = torch.min(
                torch.stack(
                    [network_output_dict["static_logit"], network_output_dict["dynamic_logit"]],
                    dim=0,
                ),
                dim=0,
                keepdim=False,
            )[0] - 100.0 * ones
        else:
            raise ValueError("unknown output mode: %s" % str(self.ground_logit))
        # #endregion ground_logit
        return network_output_dict

    def _masking_nonfilled_pillars(self, network_output_dict, filled_pillar_mask):
        """
        Masking non-filled pillars for flow and logits based on params:
        - overwrite_non_filled_pillars_with_default_flow
        - overwrite_non_filled_pillars_with_default_logits
        """

        # Choosing the mask for unfilled pillars.
        # Mask non-filled pillars
        default_values_for_nonfilled_pillars = {
            "disappearing_logit": -100.0,
            "static_logit": -100.0 if self.static_logit is False else 0.0,
            "dynamic_logit": 0.0 if self.dynamic_logit is True else -100.0,
            "ground_logit": 0.0 if self.ground_logit is True else -100.0,
            "static_flow": 0.0,
            "dynamic_flow": 0.0,
            "static_aggr_flow": 0.0,
            "static_gt_flow": 0.0,
        }

        # Dict for choosing if the non filled pillar should be filled with default value
        # These tensors won't be masked
        # modification_taboo_keys = ["weights", "staticness", "groundness", "dynamicness", "static_aggr_trans_matrix",
        #                           "class_logits", "class_probs", "class_prediction"]

        # if not self.overwrite_non_filled_pillars_with_default_flow:
        #    modification_taboo_keys += ["static_flow", "dynamic_flow", "static_aggr_flow", "gt_static_flow"]

        # if not self.overwrite_non_filled_pillars_with_default_logits:
        #    modification_taboo_keys += ["disappearing_logit", "static_logit", "dynamic_logit", "ground_logit"]

        # We are filling empty pillars with default values
        for name, default_value in default_values_for_nonfilled_pillars.items():
            # if k in modification_taboo_keys:
            #    continue

            network_output_dict[name] = torch.where(filled_pillar_mask, network_output_dict[name], default_value)

        return network_output_dict

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
        is_dynamic = network_output_dict["dynamicness"] >= dynamicness_threshold
        # STATIC - (staticness > groundness) and not(is_dynamic)
        is_static = torch.logical_and((network_output_dict["staticness"] >= network_output_dict["groundness"]),
                                      torch.logical_not(is_dynamic))
        # GROUND
        is_ground = torch.logical_not(torch.logical_or(is_static, is_dynamic))

        # Class prediction in one-hot encoding
        network_output_dict["class_prediction"] = torch.cat([is_static, is_dynamic, is_ground], dim=1)
        network_output_dict["is_static"] = is_static
        return network_output_dict

    def _procces_weights_static_aggregation(self, network_output_dict, filled_pillar_mask):

        # static_aggr_weight_map = probs of static * bool mask of filled pillars
        masked_staticness = network_output_dict["staticness"] * filled_pillar_mask[:, 0]

        mode = self.predict_weight_for_static_aggregation
        assert mode in {"sigmoid", "softmax"}
        if mode == "softmax":
            raise NotImplementedError()
        else:
            grid_size = filled_pillar_mask.shape[2:]
            prod_size = grid_size[0] * grid_size[1]
            masked_weights_for_static_aggregation = normalized_sigmoid_sum(
                logits=torch.reshape(network_output_dict["weights"], [-1, prod_size]),
                mask=torch.reshape(filled_pillar_mask, [-1, prod_size]),
            )
            masked_weights_for_static_aggregation = masked_weights_for_static_aggregation.reshape([-1, *grid_size])

            #sigmoid_weights = torch.sigmoid(network_output_dict["weights"])

        static_aggr_weight_map = (masked_staticness * masked_weights_for_static_aggregation)
        #static_aggr_weight_map = (masked_staticness * sigmoid_weights)
        return network_output_dict, static_aggr_weight_map.float()


if __name__ == "__main__":
    # for debug purposes
    test_input = [torch.rand((1, 9, 640, 640)) for x in range(6)]
    model = OutputDecoder()

    prediction_fw = model(network_output=test_input[0],
                          dynamicness_threshold=0.5,
                          pc=torch.rand((1, 95440, 5)),
                          pointwise_voxel_coordinates_fs=torch.randint(0, 640, (1, 95440, 2)),
                          pointwise_valid_mask=torch.randint(0, 2, (1, 95440)).type(torch.bool),
                          filled_pillar_mask=torch.randint(0, 2, (1, 1, 640, 640)).type(torch.bool),
                          odom=torch.rand((1, 4, 4)),
                          inv_odom=torch.rand((1, 4, 4)))
