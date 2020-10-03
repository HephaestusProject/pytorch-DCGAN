import pytest
from argparse import ArgumentParser, Namespace
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pathlib import Path

import main


@pytest.fixture(scope="module")
def model_args():
    args = Namespace(dataset="svhn", model="dcgan", experiment=None)
    return args


@pytest.fixture(scope="module")
def experiment_args():
    args = Namespace(dataset="svhn", model="dcgan", experiment="exp_1")
    return args


# config
def test_get_config(model_args, experiment_args):
    fixed_model_conf = OmegaConf.create(
        {
            "dataset": {
                "name": "svhn",
                "path": {"train": "dataset/svhn/train", "test": "dataset/svhn/test"},
                "params": {"height": 32, "width": 32, "validation_size": 10000},
            },
            "model": {"name": "dcgan"},
            "experiment": {"name": "batch_size_64"},
            "hparams": {"batch_size": 64},
        }
    )
    fixed_exp_conf = OmegaConf.create(
        {
            "dataset": {
                "name": "svhn",
                "path": {"train": "dataset/svhn/train", "test": "dataset/svhn/test"},
                "params": {"height": 32, "width": 32, "validation_size": 10000},
            },
            "model": {"name": "dcgan"},
            "experiment": {"name": "exp_1"},
            "hparams": {"batch_size": 128},
        }
    )
    conf_path = Path("tests/fixture/conf")
    model_conf = main.get_config(model_args, conf_path)

    assert type(model_conf) == DictConfig
    assert model_conf == fixed_model_conf
    exp_conf = main.get_config(experiment_args, conf_path)
    assert exp_conf == fixed_exp_conf


"""
def test_run():
    args = Namespace(dataset="svhn", model="dcgan", experiment=None, mode="train")
    main.run(args)
    assert False == True
"""
