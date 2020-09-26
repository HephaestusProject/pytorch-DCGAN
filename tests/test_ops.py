import pytest
import torch
from src.model.ops import Reshape


def test_reshape():
    x = torch.empty(128, 100)
    y = Reshape(-1, 1).forward(x)
    assert y.size() == torch.Size([12800, 1])
