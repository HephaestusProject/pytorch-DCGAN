import dataclasses
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.dataset import SVHN
from src.model.net import Discriminator, Generator
from src.runner import DCGAN, SaveCheckpointEveryNEpoch

import functools


def get_dataloader(conf: str) -> (DataLoader, DataLoader):
    if conf.dataset.name == "svhn":
        svhn = SVHN(
            train_path=conf.dataset.path.train,
            test_path=conf.dataset.path.test,
            validation_size=conf.dataset.params.validation_size,
            batch_size=conf.hparams.batch_size,
        )
        return svhn.train_dataloader(), svhn.val_dataloader()
    else:
        raise Exception(f"Invalid dataset name: {conf.dataset.name}")


def train(conf: DictConfig) -> None:
    exp_name = conf.experiment.name
    checkpoint_path = Path("checkpoints")
    wandb_logger = WandbLogger(
        name=exp_name, project="hephaestusproject-pytorch-dcgan", log_model=True,
    )
    train_dataloader, val_dataloader = get_dataloader(conf)
    model_G = Generator(hparams=conf.hparams)
    model_D = Discriminator(hparams=conf.hparams)
    runner = DCGAN(conf.hparams, model_G, model_D)
    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=-1 if torch.cuda.is_available() else 0,
        max_epochs=conf.hparams.max_epochs,
        callbacks=[
            SaveCheckpointEveryNEpoch(
                n=2, file_path=checkpoint_path, filename_prefix=exp_name
            )
        ],
    )
    trainer.fit(runner, train_dataloader=train_dataloader)
