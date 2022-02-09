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
                 robust_method: str = "forward") -> None:
        super(RobustCrossEntropyLoss, self).__init__()
        self.robust_method = robust_method
        assert self.robust_method in ["forward", "backward"]
        if self.robust_method == "forward":
            self.register_buffer("T", torch.linalg.inv(T))
        else:
            raise NotImplementedError("Backward Type is not implemented yet")


    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if self.robust_method == "forward":
            target = torch.inner(target.type(self.T.dtype), self.T)
            return - torch.mean(torch.sum(target * torch.log(pred.softmax(-1)), axis = -1))
        else:
            raise NotImplementedError("Robust Type is not implemented")