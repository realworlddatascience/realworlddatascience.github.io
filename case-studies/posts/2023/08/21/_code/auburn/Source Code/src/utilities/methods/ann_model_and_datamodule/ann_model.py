# ann.py
# <Explanation of ANN>
# Auburn Big Data

from pathlib import Path
import argparse
from abc import ABC
from typing import Tuple, List, Dict

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as func
import torch.optim as optim
from pytorch_lightning.utilities.types import (
    EPOCH_OUTPUT
)
from torch import nn

from src.utilities.user_types import DF, ArgParser
from src.utilities.official_eval.evaluation_script import evaluate


class FFNet(nn.Module):
    # Constructor
    def __init__(self, dim, ec_classes):
        super(FFNet, self).__init__()
        self.dim = dim
        self.numFC = 0
        self.numReLU = 0
        self.ec_classes = ec_classes

        # Init the model
        print(f"ANN-INIT: Initializing neural network with dimension {dim}")
        i = 0
        for i in range(len(dim) - 1):
            setattr(self, f"fc{i + 1}", nn.Linear(dim[i], dim[i + 1]))
            # print(f"ANN-INIT: Initaialized fully-connected layer {i + 1}")
            self.numFC += 1
            if i < len(dim) - 2:
                setattr(self, f"relu{self.numReLU + 1}", nn.ReLU())
                # print(f"ANN-INIT: Initaialized ReLU {i + 1}")
                self.numReLU += 1
        # print(f"ANN-INIT: Initialized {self.numFC} fully-connected layers and {self.numReLU} ReLU units")

    # Forward Pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float()
        for i in range(1, self.numFC + 1):
            out = getattr(self, f"fc{i}")(out)
            if i < self.numFC:
                out = getattr(self, f"relu{i}")(out)
        return out  # we'll be using log softmax since that is more stable  # New
        # return func.softmax(out, dim=-1)  # Old (dim=0) initially which was wrong


class ANNModel(pl.LightningModule, ABC):
    logs_keys = ["loss", "ndcg", "success"]  # TODO: chnage reward to metric name

    def __init__(
            self,
            dim,
            ec_classes,
            config: dict,
            ground_truth_loc: Path,
            val_metric: str = "ndcg",
            *args,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = FFNet(dim, ec_classes)
        self.loss_func = nn.NLLLoss()
        assert val_metric in ANNModel.logs_keys
        self.val_metric = val_metric
        self.ground_truth = self._read_ground_truth()

    def forward(self, batch_x: torch.Tensor) -> torch.Tensor:
        return self.model(batch_x)

    def training_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor, List[str]],
            batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        output = self(batch[0])
        output = func.log_softmax(output, dim=-1)  # log softmax
        loss = self.loss_func(output, batch[1])
        print(f"ANN-TRAIN: "
              f"Batch Index: {batch_idx}/{self.trainer.num_training_batches}, "
              f"Epoch: {self.current_epoch}/{self.hparams.max_epochs}, "
              f"Loss: {loss.item()}")
        return {"loss": loss}

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        outputs = torch.mean(torch.stack([ele["loss"] for ele in outputs]))
        print(f"ANN-TRAIN: "
              f"Epoch: {self.current_epoch}/{self.hparams.max_epochs}, "
              f"Loss: {outputs.item()}")

    def validation_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor, List[str]],
            batch_idx: int,
            verbose: bool = True,
    ) -> Dict[str, List[DF]]:
        if verbose:
            print(f"Validation: Batch_Idx: {batch_idx}")
        ppc_rankings = []
        output = self(batch[0])
        output = func.softmax(output, dim=-1)
        for out, upc in zip(output, batch[2]):
            ranking = pd.DataFrame({
                'upc': len(out) * [upc],
                'ec': self.hparams.ec_classes,
                'confidence': out.cpu().numpy(),
            })
            ranking.sort_values(by="confidence", ascending=False, inplace=True, ignore_index=True)
            ppc_rankings.append(ranking.head(5))

        return {"ppc_rankings": ppc_rankings}

    def _read_ground_truth(self) -> DF:
        ground_truth = pd.read_csv(self.hparams.ground_truth_loc, dtype={"upc": str, "ec": str}, keep_default_na=False)
        ground_truth.drop(columns=ground_truth.columns[2:], inplace=True)
        return ground_truth

    def get_validation_epoch_end_scores(self, outputs: EPOCH_OUTPUT) -> Tuple[float, float]:
        outputs = sum([ele["ppc_rankings"] for ele in outputs], [])
        outputs = pd.concat(outputs).reset_index(drop=True)
        # assert len(outputs) == 5*len(ground_truth)
        ndcg, success = evaluate(outputs, self.ground_truth)
        return ndcg, success

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        ndcg, success = self.get_validation_epoch_end_scores(outputs)
        self.log(f"val_ndcg", ndcg)
        self.log(f"val_success", success)

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias"]
        optimizer_grouped_params = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optim.AdamW(optimizer_grouped_params,
                                lr=self.hparams.lr)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser: ArgParser) -> ArgParser:
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--train_batch_size",
            type=int,
            help="Train Batch Size",
            default=128,
        )
        parser.add_argument(
            "--eval_batch_size",
            type=int,
            help="Eval Batch Size",
            default=128,
        )
        parser.add_argument(
            "--lr",
            default=1e-5,
            type=int,
            help="Learning rate for the model",
        )
        parser.add_argument(
            "--max_grad_norm",
            dest="gradient_clip_val",
            default=1.0,
            type=float,
            help="Max grad norm",
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            dest="accumulate_grad_batches",
            default=1,
            help="Number of update steps to accumulate before performing a backward/update pass",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Weight Decay"
        )
        parser.add_argument(
            "--early_stopping_patience",
            type=int,
            default=-1,
            required=False,
            help="-1 means never early stop. "
                 "It is measured in validation checks. "
                 "So validation check interval will affect it"
        )
        return parser





