# ann.py
# <Explanation of ANN>
# Auburn Big Data

# Internal Imports
# External Imports
import gc
import sys
import math
from time import time
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


class FFNet(nn.Module):
    # Constructor
    def __init__(self, dim, save_loc, encoder, batch_size=128):
        super(FFNet, self).__init__()
        self.dim = dim
        self.numFC = 0
        self.numReLU = 0
        self.save_loc = save_loc
        self.model_trained = False
        self.batch_size = batch_size
        self.encoder = encoder
        print(f"ANN-INIT: Initializing neural network with dimension {dim}")

        i = 0
        for i in range(len(dim) - 1):
            setattr(self, f"fc{i + 1}", nn.Linear(dim[i], dim[i + 1]))
            print(f"ANN-INIT: Initaialized fully-connected layer {i + 1}")
            self.numFC += 1
            if (i < len(dim) - 2):
                setattr(self, f"relu{self.numReLU + 1}", nn.ReLU())
                print(f"ANN-INIT: Initaialized ReLU {i + 1}")
                self.numReLU += 1
        print(f"ANN-INIT: Initialized {self.numFC} fully-connected layers and {self.numReLU} ReLU units")

    # Forward Pass
    def forward(self, x):
        out = x.float()
        for i in range(1, self.numFC + 1):
            out = getattr(self, f"fc{i}")(out)
            if (i < self.numFC):
                out = getattr(self, f"relu{i}")(out)
        return out  # we'll be using log softmax since that is more stable  # New
        # return func.softmax(out, dim=-1)  # Old (dim=0) initially which was wrong

    # Model Training
    def train_model(self, data, learn_rate=1e-4, epochs=50):
        if sys.gettrace() is not None:  # Debug Mode
            epochs = 2
        self.model_trained = False

        # loss_func = nn.CrossEntropyLoss()  # Old
        loss_func = nn.NLLLoss()  # New

        # optimizer = optim.SGD(self.parameters(), lr=learn_rate)  # Old
        optimizer = optim.Adam(self.parameters(), lr=learn_rate)  # New

        #num_iter = math.ceil(data.num_samples / self.batch_size)
        train_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True, collate_fn=data.collate_fun)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        start_time = time()
        self.to(device)
        for epoch in tqdm(range(epochs), desc=f"ANN-TRAIN: Training model - Epoch XX/{epochs}"):
            running_loss = 0
            for (input, label, _) in train_loader:
                label = label.type(torch.LongTensor)
                input = input.to(device)
                label = label.to(device)
                optimizer.zero_grad()

                input = input.view(input.shape[0], -1)
                input.float()

                output = self(input)  # logits
                output = func.log_softmax(output, dim=-1)  # log softmax

                #label.float()
                loss = loss_func(output, label)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                running_loss += loss.item()

        print(f"ANN-TRAIN: Total training time (min): {(time() - start_time) / 60}")
        state = self.state_dict()
        # print(state, type(state))
        torch.save(self, self.save_loc)
        self.model_trained = True

    # Model Inference
    def infer_data(self, data, data_loc, k=5):
        ppc_columns = [
            "upc",
            "ec",
            "confidence",
        ]
        # ppc_rankings = pd.DataFrame(columns=ppc_columns)
        ppc_rankings = []
        test_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True, collate_fn=data.collate_fun)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()
        with torch.no_grad():
            for (input, _, upc_batch) in tqdm(test_loader, desc=f"ANN-TRAIN: Evaluating batches - XX/{data.num_samples / self.batch_size}"):
                input = input.to(device)
                input = input.view(input.shape[0], -1)
                input.float()
                output = self(input)  # logits
                output = func.softmax(output, dim=-1)

                for out, upc in zip(output, upc_batch):
                    # obj = []  # each sample you get new obj
                    ranking = pd.DataFrame({
                        'upc': len(out)*[upc],
                        'ec': self.encoder.classes_,
                        'confidence': out.cpu().numpy(),
                    })
                    ranking.sort_values(by="confidence", ascending=False, inplace=True, ignore_index=True)
                    ppc_rankings.append(ranking.head(k))
                    # for i in range(k):  # Directly add df to a list and then concatenate list of dfs
                    #     obj.append({
                    #         "upc": upc,
                    #         "ec": ranking["ec"][i],
                    #         "confidence": ranking["confidence"][i],
                    #     })
                    # ppc_rankings = ppc_rankings.append(obj, ignore_index=True)

        aa = 1
        ppc_rankings = pd.concat(ppc_rankings)
        save_loc = f"./data/ann/predictions.csv"  # TODO: change based on embedding name
        ppc_rankings.to_csv(save_loc, index=False)

        # Evaluate the results
        print("RF-EVAL: Reading prediction rankings & ground truth")
        truth_loc = data_loc
        ground_truth = pd.read_csv(truth_loc, dtype={"upc": str, "ec": str}, keep_default_na=False)
        ground_truth.drop(columns=ground_truth.columns[2:], inplace=True)
        # print(ground_truth)
        predictions = pd.read_csv(save_loc, dtype={"upc": str, "ec": str},
                                  keep_default_na=False)
        # print(predictions)

        print("RF-EVAL: Evaluating the generated rankings")
        evaluate(predictions, ground_truth)



class CustomDataset(Dataset):
    # Constructor
    def __init__(self, data, encoder, flag: str = "train"):
        # Either read in the data from train.csv or just use it passed as-is
        self.encoder = encoder  # fit has already been called

        if flag == "train":
            self.labels = self.encoder.transform(data["ec"])  # transform is both same
            # self.labels = self.encoder.fit_transform(data["ec"])  # fit and transform is both same
        else:
            self.labels = torch.ones(len(list(data["ec"])))

        self.upc = data["upc"]
        self.data = torch.tensor(data.drop(columns=["ec", "upc"]).values, dtype=torch.float32)  # this should happen in __getitem__ ideally
        self.labels = torch.as_tensor(self.labels)
        self.num_samples = self.data.shape[0]
        print(f"ANN-DATA: {self.num_samples} samples were loaded")

    # Item access
    def __getitem__(self, index):
        return self.data[index], self.labels[index], self.upc[index]

    def collate_fun(self, batch):
        data = torch.stack([ii[0] for ii in batch])
        labels = torch.stack([ii[1] for ii in batch])
        upc = [ii[2] for ii in batch]
        return data, labels, upc

    # Length
    def __len__(self):
        return self.num_samples

    def num_cols(self):
        return self.data.shape[1]


def get_fndds_map(fndds_loc: str = "./data/raw/fndds_1718.csv") -> list:
    col_list = ["food_code"]
    food_codes = pd.read_csv(fndds_loc, dtype=str, keep_default_na=False, usecols=col_list, squeeze=True)
    # food_codes = fndds_data["food_code"]
    food_codes = food_codes.append(pd.Series(["-99", "2053"]))
    return food_codes.to_list()


def train(config):
    num_threads = config["general"]["num_threads"]
    # Read in the train set
    print("ANN-TRAIN: Loading the training set")
    train_data = pd.read_csv("./data/formatted/train.csv", dtype={"upc": str, "ec": str}, keep_default_na=False)
    input_size = len(train_data.columns)-2 # Magic -2 so we can drop the upc and ec columns
    # output_size = len(get_fndds_map())
    # print(f"ANN-TRAIN: {output_size} unique FNDDS classes identified for classification")

    # Prep the data
    encoder = preprocessing.LabelEncoder()
    encoder = encoder.fit(train_data["ec"])
    print("ANN-TRAIN: Preparing the training set")
    train_set = CustomDataset(train_data, encoder)
    del train_data
    gc.collect()

    output_size = len(train_set.encoder.classes_)
    # Train the model
    #x = train_data.drop(columns="ec")
    #y = train_data["ec"]
    print("ANN-TRAIN: Training the model")
    model = FFNet(
        [
            input_size,
            math.ceil((input_size - output_size) * .75),
            math.floor((input_size - output_size) * .25),
            output_size,
        ],
        config["method"]["model_loc"] + "ann/ann_model.pt",
        encoder,
    )
    model.train_model(train_set)


def infer(config, data_loc="", model_loc=""):
    #CHUNK_SIZE = 10000
    # Load the model
    print("ANN-INFER: Loading the model")
    model = (torch.load(config["method"]["model_loc"] + "ann/ann_model.pt") if not model_loc else torch.load(model_loc))

    # Read in the specified dataset
    print("ANN-INFER: Reading the provided dataset")
    test_data = pd.read_csv(data_loc, dtype={"upc": str, "ec": str}, keep_default_na=False)
    print("ANN-TRAIN: Preparing the provided set")
    test_set = CustomDataset(test_data, model.encoder, flag="test")
    del test_data
    gc.collect()

    # Run the inference function
    data_loc = "./data/formatted/test.csv" if not data_loc else data_loc
    model.infer_data(test_set, data_loc)
