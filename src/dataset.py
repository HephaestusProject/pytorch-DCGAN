import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


class SVHN:
    def __init__(self, conf):
        self.conf = conf.dataset
        self.model_params = conf.model.params
        self.dataset = self.load_dataset()
        self.validation_dataset, self.train_dataset = self.split_dataset()

    def load_dataset(self):
        return torchvision.datasets.SVHN(
            self.conf.path.train,
            split="train",
            download=True,
            transform=transforms.ToTensor(),
        )

    def split_dataset(self):
        # split by fixed validation size
        return random_split(
            self.dataset,
            [
                self.conf.params.validation_size,
                len(self.dataset) - self.conf.params.validation_size,
            ],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.model_params.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validation_dataset,
            batch_size=self.model_params.batch_size,
            shuffle=True,
        )
