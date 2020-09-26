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


def get_config(args: Namespace) -> DictConfig:
    config_dir = Path("conf")
    dataset_config_filename = f"{config_dir}/dataset/{args.dataset}.yml"
    model_config_filename = f"{config_dir}/model/{args.model}.yml"

    return OmegaConf.merge(
        {"dataset": OmegaConf.load(dataset_config_filename)},
        {"model": OmegaConf.load(model_config_filename)},
    )


def get_dataloader(conf: str) -> (DataLoader, DataLoader):
    if conf.dataset.name == "svhn":
        svhn = SVHN(conf)
        return svhn.train_dataloader(), svhn.val_dataloader()
    else:
        raise Exception(f"Invalid dataset name: {conf.dataset.name}")


import functools


def get_exp_name(params: dict) -> str:
    param_str_list = [f"{k}_{v}" for k, v in params.items()]
    name = functools.reduce(lambda first, second: first + "-" + second, param_str_list)
    return name


def run(conf: DictConfig) -> None:
    print(conf.model.params)
    exp_name = get_exp_name(conf.model.params)
    wandb_logger = WandbLogger(
        name=exp_name,
        project="hephaestusproject-pytorch-dcgan",
        log_model=True,
    )
    train_dataloader, val_dataloader = get_dataloader(conf)
    model_G = Generator(hparams=conf.model.params)
    model_D = Discriminator(hparams=conf.model.params)
    runner = DCGAN(conf.model.params, model_G, model_D)
    checkpoint_path = Path("checkpoints")
    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=-1 if torch.cuda.is_available() else 0,
        max_epochs=conf.model.params.max_epochs,
        callbacks=[
            SaveCheckpointEveryNEpoch(
                n=2, file_path=checkpoint_path, filename_prefix=exp_name
            )
        ],
    )
    trainer.fit(runner, train_dataloader=train_dataloader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="svhn", type=str)
    parser.add_argument("--model", default="dcgan", type=str)
    args = parser.parse_args()
    run(get_config(args))
