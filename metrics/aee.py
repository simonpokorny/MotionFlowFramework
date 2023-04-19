from torchmetrics import Metric
import torch

class AEE(Metric):
    """
    Calculates the Average Endpoint Error (AEE) between two flows. The AEE is the mean L2 distance between each endpoint
    of the predicted flow and the corresponding endpoint of the ground truth flow.

    """
    def __init__(self):
        super().__init__()
        self.add_state("errorL2", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, flow: torch.Tensor, gt_flow: torch.Tensor):
        """
        Updates the accumulated L2 distance error and total number of elements with the given flows.

        Args:
            flow (torch.Tensor): The predicted flow, with shape [BS, N, 3].
            gt_flow (torch.Tensor): The ground truth flow, with shape [BS, N, 3].
        """
        assert flow.shape == gt_flow.shape, f"Predicted flow have different shape in comparison with gt flow"
        err = torch.linalg.vector_norm((gt_flow - flow), ord=2, dim=2)

        self.errorL2 += err.sum().long()
        self.total += gt_flow.shape[1]

    def compute(self):
        """
        Computes the Average Endpoint Error (AEE) as the ratio of the accumulated L2 distance error and the total number
        of elements.

        Returns:
            The AEE as a float32 tensor.
        """
        return self.errorL2.float() / self.total