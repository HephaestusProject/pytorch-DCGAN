import numpy as np
import torch.nn as nn
import torch

from src.model.ops import *
import math


def conv_out_size(size, stride: int = 2) -> int:
    return int(math.ceil(float(size) / float(stride)))


class Generator(nn.Module):
    def __init__(self, hparams: dict, output_size: int = 32) -> None:
        super(Generator, self).__init__()
        self.hparams = hparams
        self.output_size = output_size

        self._model = self._build_model()

    def _build_model(self) -> nn.Sequential:

        s_2 = conv_out_size(self.output_size)
        s_4 = conv_out_size(s_2, 2)
        s_8 = conv_out_size(s_4, 2)
        s_16 = conv_out_size(s_8, 2)
        return nn.Sequential(
            # project and reshape
            nn.Linear(
                self.hparams.size_of_latent_vector,
                self.hparams.num_gen_filters * 8 * s_16 * s_16,
                bias=False,
            ),
            Reshape(-1, self.hparams.num_gen_filters * 8, s_16, s_16),
            nn.ReLU(True),
            self._conv_transpose(
                self.hparams.num_gen_filters * 8, self.hparams.num_gen_filters * 4,
            ),  # [128, 256, 4, 4]
            self._conv_transpose(
                self.hparams.num_gen_filters * 4, self.hparams.num_gen_filters * 2,
            ),  # input[128, 128, 8, 8]
            self._conv_transpose(
                self.hparams.num_gen_filters * 2, self.hparams.num_gen_filters * 1,
            ),  # input[128, 64, 16, 16]
            nn.ConvTranspose2d(
                self.hparams.num_gen_filters * 1,
                self.hparams.num_channels,
                5,
                2,
                padding=2,
                output_padding=1,
            ),
            nn.Tanh(),  # [128, 3, 32, 32]
        )

    def _conv_transpose(
        self, c_in: int, c_out: int, kernel_size: int = 5, stride: int = 2
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(
                c_in, c_out, kernel_size, stride, padding=2, output_padding=1,
            ),
            nn.BatchNorm2d(c_out),
            nn.ReLU(True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        d = self._model(z)
        return d


class Discriminator(nn.Module):
    def __init__(self, hparams):
        super(Discriminator, self).__init__()
        self.hparams = hparams
        self.main = nn.Sequential(
            self._conv(
                self.hparams.num_channels, self.hparams.num_d_filters
            ),  # [128, 64, 16, 16]
            self._conv(
                self.hparams.num_d_filters, self.hparams.num_d_filters * 2
            ),  # [128, 128, 8, 8]
            self._conv(
                self.hparams.num_d_filters * 2, self.hparams.num_d_filters * 4
            ),  # [128, 256, 4, 4]
            self._conv(
                self.hparams.num_d_filters * 4, self.hparams.num_d_filters * 8
            ),  # [128, 512, 2, 2]
            Reshape(-1, self.hparams.num_d_filters * 8 * 2 * 2),
            nn.Linear(self.hparams.num_d_filters * 8 * 2 * 2, 1, bias=False,),
            nn.Sigmoid(),
        )

    def _conv(
        self, c_in: int, c_out: int, kernel_size: int = 5, stride: int = 2
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size, stride, padding=2),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.main(input)
        return output
