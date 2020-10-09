import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from src.model.ops import *


class Generator(nn.Module):
    def __init__(self, hparams: dict) -> None:
        super(Generator, self).__init__()
        self.hparams = hparams
        self.layers = self.build_model()

    def build_model(self) -> nn.Sequential:
        model = nn.Sequential(
            OrderedDict(
                [
                    ("linear", self.linear_block()),
                    (
                        "conv_t1",
                        self.conv_transpose_block(
                            self.hparams.num_g_filters * 8,
                            self.hparams.num_g_filters * 4,
                        ),  # [128, 256, 4, 4]
                    ),
                    (
                        "conv_t2",
                        self.conv_transpose_block(
                            self.hparams.num_g_filters * 4,
                            self.hparams.num_g_filters * 2,
                        ),  # [128, 128, 8, 8]
                    ),
                    (
                        "conv_t3",
                        self.conv_transpose_block(
                            self.hparams.num_g_filters * 2,
                            self.hparams.num_g_filters * 1,
                        ),  # [128, 64, 16, 16]
                    ),
                    (
                        "conv_t4",
                        nn.ConvTranspose2d(
                            self.hparams.num_g_filters * 1,
                            self.hparams.num_channels,
                            4,
                            2,
                            1,
                        ),
                    ),
                    (
                        "tahn1",
                        nn.Tanh(),
                    ),  # [128, 3, 32, 32]
                ]
            )
        )
        return model

    def linear_block(self) -> nn.Sequential:
        # project and reshape
        model = nn.Sequential(
            nn.Linear(
                self.hparams.size_of_latent_vector,
                self.hparams.num_g_filters * 8 * 2 * 2,
                bias=False,
            ),
            nn.BatchNorm1d(self.hparams.num_g_filters * 8 * 2 * 2),
            nn.ReLU(True),
            Reshape(-1, self.hparams.num_g_filters * 8, 2, 2),
        )
        return model

    def conv_transpose_block(
        self, c_in: int, c_out: int, kernel_size: int = 5, stride: int = 2, padding=2
    ) -> nn.Sequential:
        model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(c_out),
            nn.ReLU(True),
        )
        return model

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.layers(z)


class Discriminator(nn.Module):
    def __init__(self, hparams):
        super(Discriminator, self).__init__()
        self.hparams = hparams
        self.layers = self.build_model()

    def build_model(self) -> nn.Sequential:
        model = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(
                            self.hparams.num_channels,
                            self.hparams.num_d_filters,
                            5,
                            2,
                            2,
                        ),
                    ),
                    (
                        "lrelu1",
                        nn.LeakyReLU(0.2, inplace=True),
                    ),
                    (
                        "conv_block_2",
                        self.conv_block(
                            self.hparams.num_d_filters, self.hparams.num_d_filters * 2
                        ),  # [128, 128, 8, 8]
                    ),
                    (
                        "conv_block_3",
                        self.conv_block(
                            self.hparams.num_d_filters * 2,
                            self.hparams.num_d_filters * 4,
                        ),  # [128, 256, 4, 4]
                    ),
                    (
                        "conv_block_4",
                        self.conv_block(
                            self.hparams.num_d_filters * 4,
                            self.hparams.num_d_filters * 8,
                        ),
                    ),
                    ("linear", self.linear_block()),
                ]
            )
        )
        return model

    def linear_block(self) -> nn.Sequential:
        # project and reshape
        model = nn.Sequential(
            Reshape(-1, self.hparams.num_d_filters * 8 * 2 * 2),
            nn.Linear(
                self.hparams.num_d_filters * 8 * 2 * 2,
                1,
                bias=False,
            ),
            nn.Sigmoid(),
        )
        return model

    def conv_block(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int = 5,
        stride: int = 2,
        padding: int = 2,
    ) -> nn.Sequential:
        model = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.2, inplace=True),
        )
        return model

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.layers(input)
        return output
