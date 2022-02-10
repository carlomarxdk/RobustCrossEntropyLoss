import torch
from torch import Tensor
import torch.nn as nn



class RobustCrossEntropyLoss(nn.Module):
    """CrossEntropyLoss class for noisy labels
    Args:
        T (Tensor): Row-Stochaistic transition matrix for the noise, shape (CxC)
        roobust_method (str): Specifies the method fo the robustness (either "forward" or "backward")
    """
    def __init__(self, T: Tensor = None,
                 robust_method: str = "backward") -> None:
        super(RobustCrossEntropyLoss, self).__init__()
        self.robust_method = robust_method
        assert self.robust_method in ["forward", "backward"]
        if self.robust_method == "backward":
            self.register_buffer("T", torch.linalg.inv(T))
        else:
            self.register_buffer("T", T)


    def forward(self, pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
        target = target.type(self.T.dtype)
        pred = torch.clamp(pred.softmax(-1), min = eps, max = 1-eps)

        if self.robust_method == "backward":
            target = torch.inner(target, self.T)
        else:
            pred = torch.inner(pred, self.T)

        pred = torch.log(pred)
        return - torch.mean(torch.sum(target * pred, axis = -1))
