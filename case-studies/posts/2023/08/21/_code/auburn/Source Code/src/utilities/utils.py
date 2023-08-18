import os
import pathlib
import pyodbc
import re
import pandas as pd
from typing import Union
from sklearn import model_selection
from .user_types import DF


def mkdir_p(inp_dir_or_path: str) -> str:
    """Give a file/dir path, makes sure that all the directories exists"""
    orig = inp_dir_or_path
    inp_dir_or_path = pathlib.Path(inp_dir_or_path).resolve()  # return full path
    if inp_dir_or_path.suffix:  # file
        inp_dir_or_path.parent.mkdir(parents=True, exist_ok=True)
    else:  # dir
        inp_dir_or_path.mkdir(parents=True, exist_ok=True)
    return str(inp_dir_or_path)


def get_connection(host: str, db: str):
    return pyodbc.connect(
        'Driver={SQL Server};'
        f'Server={host};'
        f'Database={db};'
        'Trusted_Connection=yes;'
    )


def clean_desc(string: str) -> str:
    string = string.lower()
    # Remove the following strings:
    strings = [
        "value not available",
        "not available",
        "none",
        "default gram weights"
    ]
    for item in strings:
        string = re.sub(rf'{item}', '', string)
    # Remove contractions
    string = re.sub(r"n't", " not", string)
    # Remove anything non-alphanumeric
    # string = re.sub(r'[^A-Za-z0-9\s]+', ' ', string)
    string = re.sub(r'[^A-Za-z\s]+', ' ', string)  # Now we're removing anything non-alphabet, including numbers
    # Remove any duplicate words
    string = " ".join(
        set(string.split(" ")))  # Naman: I do not think this will work for Bert since you are losing the order
    # Remove any trailing whitespace
    string = re.sub(r'\s+', ' ', string).strip()  # Naman: why not just strip?
    return string


def make_categorical(arr: Union[pd.Series, list]) -> list:
    # TODO: Make it binary (one-hot or using log number of bits)
    # Get a list of unique values in the array
    unique = pd.unique(arr).tolist()

    # For every value in the array, assign it the index of the unique item
    result = [unique.index(x) for x in arr]

    # Return the result
    return result


def split_data_in_train_val(
        data_pt: str,
        train_size: float,
        out_train_pt: str,
        out_val_pt: str,
        split_exist_msg: str = "",
        seed: int = 3250,
):
    if not (os.path.exists(out_train_pt) and os.path.exists(out_val_pt)):
        print(f"DATA-FORMAT: Splitting train/eval splits ({train_size} train/{1-train_size} eval)")
        # Read in the training set
        train_data = pd.read_csv(data_pt, dtype={"upc": str, "ec": str}, keep_default_na=False)
        # Split it in half, one for training & one for testing
        data_splits = model_selection.train_test_split(train_data, train_size=train_size,
                                                       test_size=1-train_size, random_state=seed)
        # Write them to csv files
        data_splits[0].to_csv(out_train_pt, index=False)
        data_splits[1].to_csv(out_val_pt, index=False)
        del train_data
        print("Split done!")
    else:
        print(split_exist_msg if split_exist_msg else "Splits already exists")


def split_data_tuning(config: dict, train_loc: str = ""):
    hp_train = "./data/formatted/hyperparameter_train.csv"
    hp_valid = "./data/formatted/hyperparameter_valid.csv"
    train_size = 0.5
    train_loc = train_loc if train_loc else "./data/formatted/train.csv"
    msg = f"DATA-FORMAT: Hyperparameter tuning splits have already been created. " \
          f"Using the previously-created splits"
    split_data_in_train_val(train_loc, train_size, hp_train, hp_valid, msg)


# def _split_data_tuning(config: dict, train_loc: str = ""):
#     hp_train = "./data/formatted/hyperparameter_train.csv"
#     hp_valid = "./data/formatted/hyperparameter_valid.csv"
#     if not (os.path.exists(hp_train) and os.path.exists(hp_valid)):
#         print(f"DATA-FORMAT: Further splitting training data into train/validation splits ({.5} train/{.5} test)")
#         # Read in the training set
#         train_data = pd.read_csv(("./data/formatted/train.csv" if not train_loc else train_loc), dtype={"upc": str, "ec": str}, keep_default_na=False)
#         # Split it in half, one for training & one for testing
#         data_splits = model_selection.train_test_split(train_data, train_size=.5, test_size=.5, random_state=3250)
#         # Write them to csv files
#         data_splits[0].to_csv(hp_train, index=False)
#         data_splits[1].to_csv(hp_valid, index=False)
#         del train_data
#     else:
#         print("DATA-FORMAT: Hyperparameter tuning splits have already been created. Using the previously-created splits")
