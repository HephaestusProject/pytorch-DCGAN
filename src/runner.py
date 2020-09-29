from collections import OrderedDict
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


class SaveCheckpointEveryNEpoch(pl.Callback):
    def __init__(self, file_path: str, n: int = 1, filename_prefix: str = "") -> None:
        self.n = n
        self.file_path = file_path
        self.filename_prefix = filename_prefix

    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:

        epoch = trainer.current_epoch
        if epoch % self.n == 0:

            fake_images = pl_module.forward(pl_module.global_z_for_validation)
            trainer.logger.experiment.log(
                {
                    "images": [wandb.Image(pl_module.fake_images, caption="fake")],
                    "epoch": epoch,
                }
            )

            # save models
            filename = f"{self.filename_prefix}_epoch_{epoch}.ckpt"
            ckpt_path = f"{self.file_path}/{filename}"
            torch.save(
                {
                    "generator": pl_module.generator.state_dict(),
                    "discriminator": pl_module.discriminator.state_dict(),
                },
                ckpt_path,
            )


class DCGAN(pl.LightningModule):
    def __init__(self, hparams: dict, generator, discriminator):
        super(DCGAN, self).__init__()
        self.hparams = hparams
        self.global_z_for_validation = torch.randn(
            16,
            self.hparams.size_of_latent_vector,
            device= 'cuda:0' if torch.cuda.is_available() else 'cpu',
        )

        def _weights_init(m):
            classname = m.__class__.__name__
            if classname.find("Conv") != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find("BatchNorm") != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        self.generator = generator.apply(_weights_init)
        self.discriminator = discriminator.apply(_weights_init)

    def configure_optimizers(self) -> (List, List):
        lr = self.hparams.lr

        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr)
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        # optimizer list and no lr_scheduler
        return [g_optimizer, d_optimizer], []

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

    def adversarial_loss(self, y_hat: torch.Tensor, y: torch.Tensor):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_images, _ = batch

        # optimizer index에 따른 구현
        # generator
        if optimizer_idx == 0:
            z = torch.randn(
                real_images.size(0),
                self.hparams.size_of_latent_vector,
                device=self.device,
            )
            self.fake_images = self.forward(z)
            valid = torch.ones(real_images.size(0), 1, device=self.device)

            g_loss = self.adversarial_loss(self.discriminator(self.fake_images), valid)
            tqdm_dict = {"g_loss": g_loss}
            output = {"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}

            return output

        # discriminator
        if optimizer_idx == 1:

            valid = torch.ones(real_images.size(0), 1, device=self.device)
            real_loss = self.adversarial_loss(self.discriminator(real_images), valid)

            fake = torch.zeros(real_images.size(0), 1, device=self.device)
            fake_loss = self.adversarial_loss(
                self.discriminator(self.fake_images.detach()), fake
            )

            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = {"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}

            return output
