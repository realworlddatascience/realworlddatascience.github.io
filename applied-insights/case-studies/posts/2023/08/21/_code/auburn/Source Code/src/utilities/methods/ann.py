# ann.py
# <Explanation of ANN>
# Auburn Big Data

# Internal Imports
# External Imports
import gc
import os
import sys
import math
import json
import itertools
from abc import ABC
from time import time

from pytorch_lightning.utilities.types import (
    TRAIN_DATALOADERS,
    EVAL_DATALOADERS,
    EPOCH_OUTPUT
)
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch
from torch import nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from src.utilities.official_eval.evaluation_script import evaluate
from sklearn.metrics import classification_report
from src.utilities.utils import split_data_in_train_val
from typing import Optional, Any, Tuple, List, Dict
import pytorch_lightning as pl
from enum import Enum
from pathlib import Path

from src.utilities.methods.ann_model_and_datamodule.callbacks import (
    get_early_stopping_callback,
    get_checkpoint_callback,
    ANNLoggingCallback,
)
from src.utilities.methods.ann_model_and_datamodule.ann_model import (
    FFNet,
    ANNModel,
)
from src.utilities.methods.ann_model_and_datamodule.ann_datamodule import (
    CustomDataset,
    ANNDataModule,
)
import argparse
from src.utilities.utils import mkdir_p
from src.utilities.user_types import DatasetTypes, DF, ArgNamespace, ArgParser


def get_fndds_map(fndds_loc: str = "./data/raw/fndds_1718.csv") -> list:
    col_list = ["food_code"]
    food_codes = pd.read_csv(fndds_loc, dtype=str, keep_default_na=False, usecols=col_list, squeeze=True)
    # food_codes = fndds_data["food_code"]
    food_codes = food_codes.append(pd.Series(["-99", "2053"]))
    return food_codes.to_list()


def train(
        config: dict,
        train_loc: str = "",
        val_loc: str = "",
        save_name: str = "",
        dm: Optional[ANNDataModule] = None,
        val_metric: str = "ndcg",
        args: Optional[ArgNamespace] = None,
):
    """
    In normal mode (no tuning), you only have train_loc. So we split it into train and val.
    """
    if not train_loc:
        data_pt = mkdir_p("./data/formatted/train.csv")
        train_loc = mkdir_p("./data/formatted/ann_train.csv")
        val_loc = mkdir_p("./data/formatted/ann_valid.csv")
        split_data_in_train_val(
            data_pt=data_pt,
            train_size=0.80,
            out_train_pt=train_loc,
            out_val_pt=val_loc,
        )
    assert bool(train_loc), "Somehow could not find training file! Weird"

    args_dict = vars(args)

    if not dm:
        dm = ANNDataModule(
            train_data_pt=Path(train_loc),
            eval_data_pt=Path(val_loc) if val_loc else None,
            **args_dict,
        )
        dm.setup()

    ground_truth_loc = val_loc  # "./data/formatted/test.csv" if not val_loc else val_loc
    input_size = dm.model_input_size
    output_size = len(dm.ec_classes)
    print("ANN-TRAIN: Define the model")
    dim_size = [input_size] + args.hidden_sizes + [output_size]
    model = ANNModel(
        dim=dim_size,
        ec_classes=dm.ec_classes,
        config=config,
        ground_truth_loc=Path(ground_truth_loc),
        val_metric=val_metric,
        **args_dict,
    )

    es_callback = get_early_stopping_callback(
        metric="ndcg",
        patience=args.early_stopping_patience,
    ) if args.early_stopping_patience >= 0 else None

    checkpoint_callback = get_checkpoint_callback(
        output_dir=config["method"]["model_loc"],
        filename='ann_model' if not save_name else save_name,
        metric=model.val_metric,
    )
    callback_list = [ANNLoggingCallback()]
    if es_callback:
        callback_list.append(es_callback)
    if checkpoint_callback:
        callback_list.append(checkpoint_callback)

    # Init Trainer
    pl.seed_everything(args.seed, workers=True)
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callback_list,
        deterministic=True,
        benchmark=False,
    )

    # ########################################
    # # # Tune the LR
    # # Run learning rate finder
    # lr_finder = trainer.tuner.lr_find(
    #     model,
    #     train_dataloaders=dm.train_dataloader(),
    #     val_dataloaders=dm.val_dataloader(),
    #     min_lr=1e-5,
    #     max_lr=0.1,
    #     # num_training=10000,
    #     # early_stop_threshold=None,
    # )
    #
    # # Results can be found in
    # print(lr_finder.results)
    #
    # # Plot with
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    #
    # # Pick point based on plot, or get suggestion
    # new_lr = lr_finder.suggestion()
    # print(f"New Learning rate: {new_lr}")
    #
    # # # update hparams of the model
    # # model.hparams.lr = new_lr
    #
    # exit(1)
    # ########################################

    # Train the model
    print("ANN-TRAIN: Training the model")
    trainer.fit(
        model,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )
    print("ANN Train: Training Done\n")


def infer(
        config,
        data_loc="",
        model_loc="",
        args: Optional[ArgNamespace] = None,
) -> Tuple[float, float]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the model
    print("ANN-INFER: Loading the trained model")
    model_ckpt = model_loc if model_loc else config["method"]["model_loc"] + "ann/ann_model.ckpt"
    model = ANNModel.load_from_checkpoint(model_ckpt)
    model.eval()
    model.to(device)

    # Prepare the data
    data_loc = data_loc if data_loc else config["method"]["inference_data"]
    dm = ANNDataModule(eval_data_pt=Path(data_loc), **model.hparams)
    dm.setup()

    # Inference
    out = []
    for b_idx, batch in enumerate(dm.val_dataloader()):
        ele0, ele1, ele2 = batch
        ele0 = batch[0].to(device)
        ele1 = batch[1].to(device)
        batch = (ele0, ele1, ele2)
        with torch.no_grad():
            out.append(model.validation_step(batch, b_idx, verbose=False))
    ndcg, success = model.get_validation_epoch_end_scores(out)
    return ndcg, success


def hyperparameter_tune(config, args: Optional[ArgNamespace] = None):
    # args.profiler = "simple"
    hp_log = mkdir_p(os.path.join(config['data']['path'], "ann/hyperparameter_log.txt"))
    # Read the checkpoint file
    checkpoint = {}
    ann_hp_json = os.path.join(config['data']['path'], "temp/ann_hyperparameter.json")
    if os.path.exists(ann_hp_json):
        with open(ann_hp_json, "r") as checkpoint_file:
            checkpoint = json.load(checkpoint_file)
    else:
        # Overwrite the performance log
        with open(hp_log, "w") as logfile:
            logfile.write("ANN Hyperparameter Combination Performances:\n\n")
            logfile.close()
        # Initialize the checkpoint
        checkpoint = {
            "best_model": {"ndcg": 0.0, "success": 0.0},
            "best_parameters": {},
            "remaining_combinations": [],
        }
        # Build the grid
        grid = {
            "max_epochs": [100, 300, 1000],
            "hidden_sizes": [[10000, 3000], [10000, 5000, 3000]],  # you can specify sizes here
            # "lr": [1e-5, 2e-5],  # figure this out first and then set it once and for all
            "weight_decay": [0.1, 0.3, 1.0],  # try more  values. Upto you
        }

        # Construct a list of combinations to run
        keys, vals = zip(*grid.items())
        checkpoint["remaining_combinations"] = [dict(zip(keys, v)) for v in itertools.product(*vals)]

    # For each item in the grid, train a small-scale model & test it.
    # Only keep a model if it performs higher than what is known
    while checkpoint["remaining_combinations"]:
        params = checkpoint["remaining_combinations"][0]
        print(f"\nRunning Params: {params}\n")
        for key, val in params.items():
            exec(f"args.{key} = {val}")

        if sys.gettrace() is not None:  # Debug mode
            args.gpus = 1
            args.default_root_dir = "./tmp"
            args.num_sanity_val_steps = 0
            args.max_epochs = 1
            args.train_batch_size = 10
            args.eval_batch_size = 1000
            args.detect_anomaly = True
            args.profiler = "simple"
            args.stochastic_weight_avg = True
            args.enable_progress_bar = False
            args.limit_train_batches = 2
            args.limit_val_batches = 2
            import warnings
            warnings.filterwarnings("ignore")

        # Load the data upfront so that you don't have to do it repeatedly
        train_loc = "./data/formatted/hyperparameter_train.csv"
        val_loc = "./data/formatted/hyperparameter_valid.csv"
        args_dict = vars(args)
        dm = ANNDataModule(
            train_data_pt=Path(train_loc),
            eval_data_pt=Path(val_loc),
            **args_dict,
        )
        dm.setup()

        # Train the model
        train(
            config,
            save_name="ann_hyperparameter",
            train_loc=train_loc,
            val_loc=val_loc,
            dm=dm,
            val_metric="ndcg",  # You can pass "success" as well
            args=args,
        )
        # Evaluate the model
        # TODO: My code needs to return something
        ndcg, success = infer(
            config,
            data_loc="./data/formatted/hyperparameter_valid.csv",
            model_loc=config["method"]["model_loc"] + "ann/ann_hyperparameter.ckpt",
            args=args,
        )
        # Log performance
        with open(hp_log, "a") as logfile:
            logfile.write(
                f"Parameters: {json.dumps(checkpoint['remaining_combinations'][0])}\n\tNDCG@5:{ndcg}\n\tSuccess@5:{success}\n\n")

        # If the current model outperforms previous models, save it
        if (ndcg > checkpoint["best_model"]["ndcg"]) and (success > checkpoint["best_model"]["success"]):
            checkpoint["best_model"]["ndcg"] = ndcg
            checkpoint["best_model"]["success"] = success
            checkpoint["best_parameters"] = checkpoint["remaining_combinations"][0]

        # Remove the hyperparameter combination & write the checkpoint
        checkpoint["remaining_combinations"].pop(0)
        with open(ann_hp_json, "w") as checkpoint_file:
            json.dump(checkpoint, checkpoint_file)
