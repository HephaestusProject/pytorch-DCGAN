import dataclasses
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from argparse import ArgumentParser, Namespace
from omegaconf import OmegaConf, DictConfig
import dataclasses
from src.dataset import SVHN
from src.model.net import Generator, Discriminator
from src.runner import Runner
import pytorch_lightning as pl


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
        raise Exception("Invalid dataset name: {}".format(conf.dataset.name))


def run(conf: DictConfig) -> None:
    train_dataloader, val_dataloader = get_dataloader(conf)
    model_G = Generator(hparams=conf.model.params)
    model_D = Discriminator(hparams=conf.model.params)
    runner = Runner(conf.model.params, model_G, model_D)
    trainer = pl.Trainer()
    trainer.fit(runner, train_dataloader=train_dataloader)
    # , val_dataloaders=val_dataloader


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="svhn", type=str)
    parser.add_argument("--model", default="dcgan", type=str)
    args = parser.parse_args()
    run(get_config(args))
