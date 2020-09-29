import pytest
import torch
from omegaconf import OmegaConf

from src.model.net import Discriminator, Generator


@pytest.fixture(scope="module")
def hparams():
    params = OmegaConf.create(
        {
            "batch_size": 128,
            "output_height": 32,
            "output_width": 32,
            "num_channels": 3,
            "size_of_latent_vector": 100,
            "num_g_filters": 64,
            "num_d_filters": 64,
            "lr": 0.0002,
            "beta1": 0.5,
            "max_epochs": 500,
        }
    )
    return params


def test_discriminator(hparams):
    x = torch.empty(128, 3, 32, 32)
    y = Discriminator(hparams).forward(x)
    assert y.size() == torch.Size([128, 1])


def test_generator(hparams):
    z = torch.empty(128, 100)
    y = Generator(hparams).forward(z)
    assert y.size() == torch.Size([128, 3, 32, 32])
