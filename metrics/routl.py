from torchmetrics import Metric
import torch

class ROutl(Metric):
    """
    Calculates the point ratio where the endpoint error (EE) and the relative error is bigger than 0.3.

    The endpoint error (EE) is the L2 distance between each endpoint of the predicted flow and the corresponding
    endpoint of the ground truth flow. The relative error is the EE divided by the L2 norm of the ground truth flow.
    The AccS is the ratio of the number of points where the EE and the relative error is higher than 0.3 and
    the total number of points in the ground truth flow.
    """
    def __init__(self):
        super().__init__()
        self.add_state("outliers", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, flow: torch.Tensor, gt_flow: torch.Tensor):
        """
        Args:
            flow (torch.Tensor): The predicted flow, with shape [BS, N, 3].
            gt_flow (torch.Tensor): The ground truth flow, with shape [BS, N, 3].
        """
        assert flow.shape == gt_flow.shape, f"Predicted flow have different shape in comparison with gt flow"

        # Computing the error only in xy coordinates
        err = torch.linalg.vector_norm(((gt_flow - flow))[:, :, :2], ord=2, dim=2)
        relative_err = err / torch.linalg.vector_norm(gt_flow[:, :, :2], ord=2, dim=2)

        self.outliers += torch.logical_and(err > 0.3, relative_err > 0.3).sum()
        self.total += gt_flow.shape[1]

    def compute(self):
        """
        Returns:
            The AccS as a float32 tensor.
        """
        return self.outliers.float() / self.total

