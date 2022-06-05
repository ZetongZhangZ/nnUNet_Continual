from torch import nn, Tensor
import torch

class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        loss = super().forward(input, target.long())
        if self.reduction == 'none':
            loss = torch.mean(loss,dim = (1,2,3))
        return loss