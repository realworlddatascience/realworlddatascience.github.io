import gc
import os
import sys
from abc import ABC
from pathlib import Path
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import (
    TRAIN_DATALOADERS,
    EVAL_DATALOADERS
)
from sklearn import preprocessing
from sklearn import model_selection
from torch.utils.data import Dataset, DataLoader

from src.utilities.user_types import DatasetTypes


class CustomDataset(Dataset):
    # Constructor
    def __init__(self, data, encoder=None, flag: str = "train"):
        # Either read in the data from train.csv or just use it passed as-is
        self.encoder = encoder  # fit has already been called

        if flag == "train":
            assert encoder is not None, "You need to pass encoder in case of training data"
            self.labels = self.encoder.transform(data["ec"])  # transform is both same
            # self.labels = self.encoder.fit_transform(data["ec"])  # fit and transform is both same
        else:
            self.labels = torch.ones(len(list(data["ec"])))

        self.upc = data["upc"]
        self.data = torch.tensor(data.drop(columns=["ec", "upc"]).values,
                                 dtype=torch.float32)  # this should happen in __getitem__ ideally
        self.labels = torch.as_tensor(self.labels)
        self.num_samples = self.data.shape[0]
        print(f"ANN-DATA: {self.num_samples} samples were loaded")

    # Item access
    def __getitem__(self, index):
        return self.data[index], self.labels[index], self.upc[index]

    @staticmethod
    def collate_fun(batch):
        data = torch.stack([ii[0] for ii in batch])
        labels = torch.stack([ii[1] for ii in batch])
        upc = [ii[2] for ii in batch]
        return data, labels.type(torch.LongTensor), upc

    # Length
    def __len__(self):
        return self.num_samples

    def num_cols(self):
        return self.data.shape[1]


class ANNDataModule(pl.LightningDataModule, ABC):
    def __init__(
            self,
            train_data_pt: Optional[Path] = None,
            eval_data_pt: Optional[Path] = None,
            *args,
            **kwargs
    ):
        super().__init__()
        assert not (train_data_pt is None and eval_data_pt is None), "Both train and eval paths cant be None"
        self.save_hyperparameters(logger=False)  # all params are accessible via self.hparams

        self.train_set = None
        self.eval_set = None
        self.ec_classes = None
        self.model_input_size = None

        self.num_workers = os.cpu_count()
        self.pin_memory = True

    def setup(self, stage: Optional[str] = None) -> None:
        """
        We have three possibilities:
        1. Train Data Path is None. Means we are doing inference. Load the validation data/dataloader
        2. Eval Data Path is None. Means training. Divide train data in train and val and create loaders
        3. Both not None. Create loaders from respective paths
        """
        nrows = None
        # if sys.gettrace() is not None:  # Debug mode
        #     nrows = 1000

        if self.hparams.train_data_pt:
            print("ANN-TRAIN: Loading the training set")
            train_data = pd.read_csv(
                self.hparams.train_data_pt,
                dtype={"upc": str, "ec": str},
                keep_default_na=False,
                nrows=nrows,
            )
            self.model_input_size = len(train_data.columns) - 2  # Magic -2 so we can drop the upc and ec columns
            encoder = preprocessing.LabelEncoder()
            encoder = encoder.fit(train_data["ec"])
            self.train_set = CustomDataset(train_data, encoder)
            self.ec_classes = self.train_set.encoder.classes_
            del train_data

        if self.hparams.eval_data_pt:
            print("ANN-INFER: Reading the validation dataset")
            val_data = pd.read_csv(
                self.hparams.eval_data_pt,
                dtype={"upc": str, "ec": str},
                keep_default_na=False,
                nrows=nrows,
            )
            self.eval_set = CustomDataset(val_data, encoder=None, flag="test")
            print("DataModule Setup Done!")
            del val_data

        gc.collect()

    def get_dataloader(self, type_path: DatasetTypes, batch_size: int, shuffle: bool = False) -> DataLoader:
        if type_path.value == "train":
            dataset = self.train_set
        elif type_path.value == "eval":
            dataset = self.eval_set
        else:
            dataset = None
            print("Not implemented")
            exit(1)

        if sys.gettrace() is not None:  # Debug Mode
            self.num_workers = 0
            self.pin_memory = False

        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=CustomDataset.collate_fun,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.get_dataloader(
            DatasetTypes.Train,
            batch_size=self.hparams.train_batch_size,
            shuffle=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.get_dataloader(
            DatasetTypes.Eval,
            batch_size=self.hparams.eval_batch_size,
            shuffle=False
        )




