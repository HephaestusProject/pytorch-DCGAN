import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import wandb


class Runner(pl.LightningModule):
    def __init__(self, hparams: dict, generator, discriminator):
        super(Runner, self).__init__()
        self.hparams = hparams
        self.generator = generator
        self.discriminator = discriminator

    def configure_optimizers(self):
        lr = self.hparams.lr

        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr)
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [g_optimizer, d_optimizer], []  # optimizer list 전달

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

    def adversarial_loss(self, y_hat: torch.Tensor, y: torch.Tensor):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_images, labels = batch

        # optimizer index에 따른 구현
        # generator
        if optimizer_idx == 0:
            z = torch.randn(real_images.size(0), self.hparams.size_of_latent_vector)

            self.fake_images = self.forward(z)
            valid = torch.ones(real_images.size(0), 1)

            g_loss = self.adversarial_loss(self.discriminator(self.fake_images), valid)
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict(
                {"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
            wandb.log({"images": [wandb.Image(self.fake_images[0], caption="fake")]})

            return output

        # discriminator
        if optimizer_idx == 1:

            valid = torch.ones(real_images.size(0), 1)
            real_loss = self.adversarial_loss(self.discriminator(real_images), valid)

            fake = torch.zeros(real_images.size(0), 1)
            fake_loss = self.adversarial_loss(
                self.discriminator(self.fake_images.detach()), fake
            )

            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict(
                {"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
            return output
