from typing import List

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split, Sampler
import numpy as np

np.random.seed(777)


class SVHN:
    def __init__(self, conf: dict) -> None:
        self.conf = conf.dataset
        self.model_params = conf.model.params
        self.train_path = self.conf.path.train
        self.test_path = "dataset/svhn/test"
        self.validation_size = self.conf.params.validation_size
        self.batch_size = self.model_params.batch_size
        self.dataset = self.load_dataset()
        self.validation_dataset, self.train_dataset = self.split_dataset()

    def load_dataset(self) -> Dataset:
        return torchvision.datasets.SVHN(
            self.train_path,
            split="train",
            download=True,
            transform=transforms.ToTensor(),
        )

    def split_dataset(self) -> [Dataset, Dataset]:
        # split by fixed validation size
        return random_split(
            self.dataset,
            [self.validation_size, len(self.dataset) - self.validation_size,],
            generator=torch.Generator().manual_seed(2147483647),
        )

    def train_dataloader(self, sampler: Sampler = None) -> Dataset:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            sampler=sampler,
        )

    def val_dataloader(self) -> Dataset:
        return DataLoader(dataset=self.validation_dataset, batch_size=self.batch_size,)

    def get_uniform_dataset_from_each_class(
        self, n: int = 1000, mode: str = "train"
    ) -> Dataset:
        # for svm training
        NUM_OF_CLASSES = 10
        dataset = None
        if mode == "train":
            dataset = self.train_dataset
        else:
            dataset = self.get_test_dataset()
        subsets = []
        for i in range(NUM_OF_CLASSES):
            sub_indices = (dataset.labels == i).nonzero()[0]
            random_indices = np.random.choice(sub_indices, n)
            subsets.extend(random_indices)
        np.random.shuffle(subsets)
        # subset = torch.utils.data.Subset(self.dataset, subsets)

        return dataset.data[subsets], dataset.labels[subsets]

    def get_test_dataset(self) -> Dataset:
        test_dataset = torchvision.datasets.SVHN(
            self.test_path, split="test", download=True,
        )
        return test_dataset

