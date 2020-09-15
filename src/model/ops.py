import torch.nn as nn
import torch


class Reshape(nn.Module):
    def __init__(self, *args) -> None:
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(self.shape)
