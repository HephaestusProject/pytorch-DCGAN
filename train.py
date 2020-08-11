import dataclasses
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import torchvision
from omegaconf import DictConfig, OmegaConf


def get_config(args: Namespace) -> DictConfig:
    config_dir = Path("conf")
    dataset_config_filename = f"{config_dir}/dataset/{args.dataset}.yml"
    model_config_filename = f"{config_dir}/model/{args.model}.yml"

    return OmegaConf.merge(
        {"dataset": OmegaConf.load(dataset_config_filename)},
        {"model": OmegaConf.load(model_config_filename)},
    )


def get_dataset(dataset: str):
    if dataset.name == "svhn":
        return torchvision.datasets.SVHN(
            dataset.path.train, split="train", download=True
        )
    else:
        raise Exception("Invalid dataset name: {}".format(dataset.name))


def run(conf: DictConfig) -> None:
    dataset = get_dataset(conf.dataset)
    validation_length = 10000
    validation_dataset, train_dataset = torch.utils.data.random_split(
        dataset,
        [validation_length, dataset.__len__() - validation_length],
        generator=torch.Generator().manual_seed(42),
    )
    validation_data_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=conf.model.params.batch_size,
        shuffle=True,
    )
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=conf.model.params.batch_size, shuffle=True,
    )

    print(validation_dataset.__len__())


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="svhn", type=str)
    parser.add_argument("--model", default="dcgan", type=str)
    args = parser.parse_args()
    run(get_config(args))
