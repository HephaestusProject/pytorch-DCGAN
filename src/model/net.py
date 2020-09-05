import numpy as np
import torch.nn as nn

from src.model.ops import *


class Generator(nn.Module):
    def __init__(self, hparams) -> None:
        super(Generator, self).__init__()
        self.hparams = hparams
        self._model = nn.Sequential(
            # project and reshape
            nn.Linear(self.hparams.size_of_latent_vector, 1024 * 4 * 4, bias=False),
            Reshape(-1, 1024, 4, 4),
            nn.ReLU(True),
            # conv1:  # 128, 512, 8, 8
            self._conv_transpose(1024, 512, 5, 1, 0),
            # conv2
            self._conv_transpose(
                self.hparams.num_gen_filters * 8,
                self.hparams.num_gen_filters * 4,
                4,
                2,
                1,
            ),
            # conv3: [128, 256, 16, 16]
            self._conv_transpose(
                self.hparams.num_gen_filters * 4,
                self.hparams.num_channels,
                4,
                2,
                1,
            ),  # [128, 3, 32, 32] # tahn으로 바꿔야함
        )

    def _conv_transpose(self, c_in, c_out, kernel_size, stride, padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(c_in, c_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(True),
        )

    def forward(self, z):
        d = self._model(z)
        return d


class Discriminator(nn.Module):
    def __init__(self, hparams):
        super(Discriminator, self).__init__()
        self.hparams = hparams
        self.main = nn.Sequential(
            self._conv(3, 32, 4, 2, 1),
            self._conv(32, 64, 4, 2, 1),
            self._conv(64, 64 * 2, 4, 2, 1),
            nn.Conv2d(64 * 2, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def _conv(self, c_in, c_out, kernel_size, stride, padding=0):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)
