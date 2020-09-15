import pytest
import pytorch_lightning
import torch

from src.model.ops import add, multiply, subtract


@pytest.fixture(scope="module")
def fix_seed():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
