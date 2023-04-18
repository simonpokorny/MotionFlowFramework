from typing import Tuple

import pytorch_lightning as pl
import torch


class MovingAverageThreshold(pl.LightningModule):
    def __init__(
            self,
            num_train_samples,
            num_points,
            resolution: int = 100000,
            start_value: float = 0.5,
            value_range: Tuple[float, float] = (0.0, 1.0)):
        """
        Args:
            num_train_samples (int): Number of training samples.
            num_points (int): Number of moving samples. In self-supervised manner, num of points in all train dataset.
            resolution (int, optional): Resolution of the moving average (resolution times different threshold)
            start_value (float, optional): Starting value for the threshold.
            value_range (Tuple[float, float], optional): Range of the threshold.

        """
        super().__init__()
        assert num_train_samples > 0, num_train_samples

        self.num_points = torch.tensor(num_points, requires_grad=False)
        value_range = torch.tensor([value_range[0], value_range[1] - value_range[0]], requires_grad=False)
        resolution = torch.tensor(resolution, requires_grad=False)
        start_value = torch.tensor(start_value, requires_grad=False)
        value_range = torch.tensor(value_range, requires_grad=False)
        avg_points_per_sample = num_points / num_train_samples

        # save variables as register buffer to save them and load in state_dict but  not to optimized them
        self.register_buffer('moving_average_importance', torch.zeros((resolution,), requires_grad=False))
        self.register_buffer('bias_counter', torch.tensor([0], requires_grad=False))

        # update buffer roughly every 5k iterations, so 5k * points per sample for denominator
        self.register_buffer('update_weight', torch.tensor(1.0 / min(2.0 * num_points, 5_000.0 * avg_points_per_sample),
                                                           requires_grad=False))
        self.register_buffer('resolution', resolution)
        self.register_buffer('start_value', start_value)
        self.register_buffer('value_range', value_range)

    def value(self):
        return torch.where(
            self.bias_counter > 0.0,
            self._compute_optimal_score_threshold(),
            self.start_value,
        )

    def _compute_bin_idxs(self, dynamicness_scores):
        idxs = torch.tensor(self.resolution * (dynamicness_scores - self.value_range[0]) / self.value_range[1],
                            dtype=torch.int32, device=self.device)

        assert (idxs <= self.resolution).all()
        assert (idxs >= 0).all()
        idxs = torch.minimum(idxs, torch.tensor(self.resolution - 1, device=self.device))
        assert (idxs < self.resolution).all()
        return idxs

    def _compute_optimal_score_threshold(self):
        improv_over_thresh = torch.cat([torch.tensor([0], device=self.device),
                                        torch.cumsum(self.moving_average_importance, dim=0)], dim=0)

        best_improv = torch.min(improv_over_thresh)
        avg_optimal_idx = torch.tensor(torch.where(best_improv == improv_over_thresh)[0],
                                       device=self.device).float().mean()

        optimal_score_threshold = (self.value_range[0] + avg_optimal_idx * self.value_range[1] / self.resolution)
        return optimal_score_threshold

    def _update_values(self, cur_value, cur_weight):

        cur_update_weight = (1.0 - self.update_weight * 100) ** cur_weight
        self.moving_average_importance = (self.moving_average_importance * cur_update_weight) + \
                                         ((1 - cur_update_weight) * cur_value)

        self.bias_counter = self.bias_counter * cur_update_weight + 1.0 - cur_update_weight

    def update(
            self,
            epes_stat_flow,
            epes_dyn_flow,
            dynamicness,
            training,
    ):
        assert isinstance(training, bool)
        if training:
            assert len(epes_stat_flow.shape) == 1
            assert len(epes_dyn_flow.shape) == 1
            assert len(dynamicness.shape) == 1
            epes_stat_flow = epes_stat_flow.detach().to(self.device)
            epes_dyn_flow = epes_dyn_flow.detach().to(self.device)
            dynamicness = dynamicness.detach().to(self.device)

            bin_idxs = self._compute_bin_idxs(dynamicness)
            improvements = epes_stat_flow - epes_dyn_flow

            # scatter nd
            cur_result = torch.zeros((self.resolution,), device=self.device)
            for i, (idx, err) in enumerate(zip(bin_idxs, improvements)):
                cur_result[idx] += err
            # end scatter nd

            self._update_values(cur_result, torch.tensor(epes_stat_flow.size(), device=self.device))

            result = self.value()
            return result
        return self.value()


if __name__ == "__main__":
    dynamicness = torch.tensor([0.1, 0.2, 0.4, 0.5, 0.6, 0.8])
    epes_stat_flow = torch.tensor([0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
    epes_dyn_flow = torch.tensor([0.6, 0.4, 0.0, 0.8, 0.4, 0.0])
    THRS = MovingAverageThreshold(
        num_train_samples=10,
        num_points=8,
        resolution=100000)
    # threshold_layer = MovingAverageThreshold(4, 8)
    for _i in range(10):
        print(THRS.value())
        opt_thresh = THRS.update(
            epes_stat_flow,
            dynamicness,
            training=True)
