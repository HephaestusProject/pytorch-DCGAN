import pytorch_lightning
import torch
from argparse import ArgumentParser
import omegaconf
import torchvision


def get_dataset(dataset: str):
    return torchvision.datasets.SVHN(split="train")


def run(args) -> None:
    # dataset 을 가져오고
    print(args.dataset)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", default="svhn", type=str, help="the name of dataset"
    )
    args = parser.parse_args()
    run(args)
