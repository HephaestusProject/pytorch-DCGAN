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

# models
import experiment_dcgan


def get_config(args: Namespace, conf_dir: str) -> DictConfig:
    exp_conf_path = f"{conf_dir}/experiment"
    model_conf_path = f"{conf_dir}/model"
    dataset_conf_path = f"{conf_dir}/dataset"

    if args.experiment:
        # overwrite model hparams with experiment params
        exp_config_filename = f"{exp_conf_path}/{args.experiment}.yml"
        experiment_config = OmegaConf.load(exp_config_filename)

        dataset_config_filename = f"{dataset_conf_path}/{experiment_config.dataset}.yml"
        dataset_config = OmegaConf.load(dataset_config_filename)

        model_config_filename = f"{model_conf_path}/{experiment_config.model}.yml"
        model_config = OmegaConf.load(model_config_filename)

        return OmegaConf.merge(
            {"dataset": dataset_config},
            {"model": {"name": model_config.name}},
            {"experiment": {"name": experiment_config.name}},
            {"hparams": experiment_config.hparams},
        )
    else:
        dataset_config_filename = f"{dataset_conf_path}/{args.dataset}.yml"
        dataset_config = OmegaConf.load(dataset_config_filename)
        model_config_filename = f"{model_conf_path}/{args.model}.yml"
        model_config = OmegaConf.load(model_config_filename)

        exp_name = get_exp_name(model_config.hparams)
        return OmegaConf.merge(
            {"dataset": OmegaConf.load(dataset_config_filename)},
            {"model": {"name": model_config.name}},
            {"experiment": {"name": exp_name}},
            {"hparams": model_config.hparams},
        )




def get_exp_name(params: dict) -> str:
    param_str_list = [f"{k}_{v}" for k, v in params.items()]
    name = functools.reduce(lambda first, second: first + "-" + second, param_str_list)
    return name


def run(args: Namespace) -> None:
    config_dir = Path("conf")
    conf = get_config(args, config_dir)
    mode = args.mode
    if mode == "train":
        if conf.model.name == "dcgan":
            experiment_dcgan.train(conf)

    # elif mode == "test":


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="svhn", type=str)
    parser.add_argument("--model", default="dcgan", type=str)
    parser.add_argument("--experiment", default=None, type=str)
    parser.add_argument("--mode", default="train", type=str)
    args = parser.parse_args()
    run(args)
