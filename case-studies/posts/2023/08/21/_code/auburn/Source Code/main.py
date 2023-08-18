# main.py
# This script acts as the entry point for the project. It takes the configuration selected and runs it (either selected inline or as a commandline parameter)
# Auburn Big Data

# Intel-CPU Patching for CPU speedups (https://intel.github.io/scikit-learn-intelex/)
from sklearnex import patch_sklearn
patch_sklearn()

import sys
import os
import json
import argparse
import pytorch_lightning as pl
from typing import Optional

from src.utilities.user_types import ArgNamespace, AvailableMethods
#from src.interface.fetch_data import main as fetch_data
from src.utilities.methods import prep_data as data_prep
from src.utilities.methods import random_forest, ann
from src.utilities.utils import split_data_tuning, mkdir_p


def get_arguments() -> ArgNamespace:
    parser = argparse.ArgumentParser(
        description="Pass lightning trainer specific arguments here",
    )
    # User Arguments
    parser.add_argument(
        "--config_path",
        help="Path to the config file",
        default="./config/default.json"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the model"
    )
    # Trainer Arguments
    parser = pl.Trainer.add_argparse_args(parser)
    # Add Model Specific Args (Don't change the order)
    parser = ann.ANNModel.add_model_specific_args(parser)
    # Parse the arguments
    return parser.parse_args()


def main(args: Optional[ArgNamespace] = None) -> None:
    # Start by verifying the selected configuration file
    config_path = args.config_path
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as config_file:
            config = json.load(config_file)
            config_file.close()
        # TODO: run config verification here later
    else:
        print(f"FATAL ERROR: The configuration file is missing. "
              f"Please specify a configuration file and/or add a default configuration file.")
        exit(1)

    # Figure out which method is being run
    active_method = config["method"]["active_method"]
    active_data = config["database"]["active_data"]

    # Assign params to args from config when active method is ann
    if active_method.lower() == AvailableMethods.ANN.value:
        args.default_root_dir = mkdir_p("./logs/ann")
        if sys.gettrace() is None:  # Normal mode
            args.logger = True
            args.profiler = "simple"
        for key, val in config['method']['method_list'][AvailableMethods.ANN.value].items():
            exec(f"args.{key} = {val}")

    # Gather the necessary data for the method
    # Condition this on having the raw data
    if not (os.path.exists(f"./data/raw/fndds_{active_data}.csv") and os.path.exists(f"./data/raw/iri_{active_data}.csv") and os.path.exists(f"./data/raw/ppc_{active_data}.csv")):
        #fetch_data(config)
        print("FATAL ERROR: The necessary data has not been supplied.")
        exit(1)

    # If the data needs to be prepared, prepare it. Otherwise, use the pre-existing prepped data
    if config["method"]["method_list"][active_method] and config["method"]["prep_data"]:
        data_prep.prep_data(config)
        print("DATA-PREP: Data preparation finished")
    else:
        print(f"DATA-PREP: The configuration indicates not to prepare the data. "
              f"Using previously-prepared data instead")

    # Hyperparameter testing will happen here
    if config["method"]["method_list"][active_method] and config["method"]["tune_hyperparameters"]:
        # Start by splitting the data
        split_data_tuning(config)
        # Run hyperparameter tuning
        methods = {
            "random_forest": random_forest.hyperparameter_tune,
            "ann": ann.hyperparameter_tune,
        }
        methods[active_method](config, args)
    else:
        print("MODEL-TUNE: The configuration indicates not to tune hyperparameters")

    # If the method needs a model, check for its existence. If it doesn't exist, train it
    if config["method"]["method_list"][active_method] and config["method"]["train_model"]:
        methods = {
            "random_forest": random_forest.train,
            "ann": ann.train
        }
        methods[active_method](config, args=args)
        print("MODEL-TRAIN: Model training finished")
    else:
        print("MODEL-TRAIN: The configuration indicates not to train a model. Using previously-trained model instead")

    # Run inference
    if config["method"]["method_list"][active_method] and config["method"]["run_inference"]:
        methods = {
            "random_forest": random_forest.infer,
            "ann": ann.infer
        }
        methods[active_method](config, data_loc=config["method"]["inference_data"], args=args)
        #random_forest.infer(config, data_loc=config["method"]["inference_data"])
        #ann.infer(config, data_loc=config["method"]["inference_data"])
        #random_forest.infer_singlethread(config, dataloc=config["method"]["method_list"][active_method]["inference_data"])
    else:
        print("MODEL-INFER: The configuration indicates not to run inference on the model")


if __name__ == '__main__':
    main(args=get_arguments())
