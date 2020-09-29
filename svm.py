import dataclasses

# memory profiling
import linecache
import os
import pickle
import tracemalloc
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pytorch_lightning as pl
import sklearn
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader

from src.dataset import SVHN
from src.model.net import Discriminator, Generator


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

    # model load and concat
    exp_name = get_exp_name(conf.model.params)
    model_D = Discriminator(hparams=conf.model.params)
    checkpoints = torch.load(
        "batch_size_128-output_height_32-output_width_32-num_channels_3-size_of_latent_vector_100-num_g_filters_64-num_d_filters_64-lr_0.0002-beta1_0.5-max_epochs_500_epoch_326.ckpt",
        map_location=torch.device("cpu"),
    )
    discriminator = model_D.load_state_dict(checkpoints["discriminator"])
    model_D.eval()

    def get_features(X):
        images = torch.from_numpy(X).type(torch.FloatTensor)
        images = images.view(-1, 3, 32, 32)
        result1 = model_D.layers.conv1.forward(images)
        result2 = model_D.layers.conv2.forward(result1)
        result3 = model_D.layers.conv3.forward(result2)
        features = (
            torch.cat(
                [
                    result1.view(-1, 64 * 16 * 16),
                    result2.view(-1, 128 * 8 * 8),
                    result3.view(-1, 256 * 4 * 4),
                ],
                axis=1,
            )
            .detach()
            .numpy()
        )
        return features

    svhn = SVHN(conf)
    X, y = svhn.get_uniform_dataset_from_each_class(mode="train")
    features = get_features(X)

    test_X, test_y = svhn.get_uniform_dataset_from_each_class(n=100, mode="test")
    test_features = get_features(test_X)
    val_X, val_y = svhn.validation_dataset.data, svhn.validation_dataset.labels
    val_features = get_features(val_X)
    # svm code
    cs = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
    for c in cs:
        print(len(X), len(y))
        clf = sklearn.svm.LinearSVC()
        clf.fit(features, y)

        # save the model to disk
        filename = f"finalized_model_{c}.sav"
        pickle.dump(clf, open(filename, "wb"))

        tr_pred = clf.predict(test_features)
        va_pred = clf.predict(val_features)

        tr_acc = sklearn.metrics.accuracy_score(test_y, tr_pred)
        va_acc = sklearn.metrics.accuracy_score(val_y, va_pred)
        print(c, tr_acc, va_acc)

        # load the model from disk
        # loaded_model = pickle.load(open(filename, 'rb'))

        ## test
        # result = loaded_model.score(X_test, Y_test)
        # print(result)

    # svm


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="svhn", type=str)
    parser.add_argument("--model", default="dcgan", type=str)
    args = parser.parse_args()
    run(get_config(args))
