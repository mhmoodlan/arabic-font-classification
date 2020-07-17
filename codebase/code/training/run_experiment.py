#!/usr/bin/env python

"""Script to run an experiment."""
import argparse
import json
import importlib
from typing import Dict
import os

from util import train_model

import numpy as np
import tensorflow as tf
import random as rn

rn.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED']=str(0)

DEFAULT_TRAIN_ARGS = {"batch_size": 32, "epochs": 10, "mode": "val", "validate_mismatch": "False"}


def run_experiment(experiment_config: Dict, save_weights: bool):
    
    experiment_config["train_args"] = {
      **DEFAULT_TRAIN_ARGS,
      **experiment_config.get("train_args", {}),
    }
    experiment_config["experiment_group"] = experiment_config.get("experiment_group", None)

    print(f"Running experiment with config {experiment_config}")
    
    datasets_module = importlib.import_module("font_classifier.datasets")
    dataset_class_ = getattr(datasets_module, experiment_config["dataset"])
    dataset_args = experiment_config.get("dataset_args", {
      'test_mode_on': True if experiment_config["train_args"]["mode"] == 'test' else False
      })
    dataset = dataset_class_(**dataset_args)
    dataset.load_or_generate_data()
    print(dataset)

    models_module = importlib.import_module("font_classifier.models")
    model_class_ = getattr(models_module, experiment_config["model"])

    networks_module = importlib.import_module("font_classifier.networks")
    network_fn_ = getattr(networks_module, experiment_config["network"])
    network_args = experiment_config.get("network_args", {})
    model = model_class_(
        dataset_cls=dataset_class_, network_fn=network_fn_, dataset_args=dataset_args, network_args=network_args,
    )
    print(model)

    train_model(
        model,
        dataset,
        epochs=experiment_config["train_args"]["epochs"],
        batch_size=experiment_config["train_args"]["batch_size"]
    )

    if experiment_config["train_args"]["validate_mismatch"] == "True":
      if experiment_config["train_args"]["mode"] == "val":
        try:
          mismatch_score = model.evaluate(dataset.mismatch_ds)
          print(f"Data mismatch score: {mismatch_score}")
        except AttributeError:
          print(f"Dataset: {dataset_class_} doesn't support mismatch validation.")
      elif experiment_config["train_args"]["mode"] == "test":
        print('In test mode, mismatch data isn\'t validated since it\'s used during training.')
    
    if experiment_config["train_args"]["mode"] == "test":
      score = model.evaluate(dataset.test_ds)
      print(f"Test score: {score}")

    if save_weights:
        model.save_weights()


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save",
        default=False,
        dest="save",
        action="store_true",
        help="If true, then final weights will be saved to canonical, version-controlled location",
    )
    parser.add_argument(
        "experiment_config",
        type=str,
        help='Experimenet JSON (\'{"dataset": "RuFaDataset", "model": "FontModel", "network": "cnn"}\'',
    )
    args = parser.parse_args()
    return args


def main():
    """Run experiment."""
    args = _parse_args()
    
    experiment_config = json.loads(args.experiment_config)
    run_experiment(experiment_config, args.save)


if __name__ == "__main__":
    main()
