from pytorch_lightning import LightningDataModule
from pandas import DataFrame
from enum import Enum
import argparse

DF = DataFrame
ArgNamespace = argparse.Namespace
ArgParser = argparse.ArgumentParser

class DatasetTypes(Enum):
    Train = "train"
    Eval = "eval"


class AvailableMethods(Enum):
    ANN = "ann"
    RF = "random_forest"
