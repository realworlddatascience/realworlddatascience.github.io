# ann.py
# <Explanation of ANN>
# Auburn Big Data

# Internal Imports
# External Imports
import math, gc
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
from sklearn.metrics import classification_report


class FFNet(nn.Module):
    # Constructor
    def __init__(self, dim, save_loc, batch_size=32):
        super(FFNet, self).__init__()
        self.dim = dim
        self.numFC = 0
        self.numReLU = 0
        self.save_loc = save_loc
        self.model_trained = False
        self.batch_size = batch_size
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
        return func.softmax(out, dim=0)

    # Model Training
    def train_model(self, data, learn_rate=.001, epochs=150):
        self.model_trained = False
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=learn_rate)
        #num_iter = math.ceil(data.num_samples / self.batch_size)
        train_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        start_time = time()
        self.to(device)
        for epoch in tqdm(range(epochs), desc=f"ANN-TRAIN: Training model - Epoch XX/{epochs}"):
            running_loss = 0
            for (input, label) in train_loader:
                label = label.type(torch.LongTensor)
                input = input.to(device)
                label = label.to(device)
                optimizer.zero_grad()

                input = input.view(input.shape[0], -1)
                input.float()

                output = self(input)

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
    def infer_data(self, data):
        ppc_columns = ["upc", "ec", "confidence"]
        ppc_rankings = pd.DataFrame(columns=ppc_columns)
        test_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()
        with torch.no_grad():
            for (input, label) in tqdm(test_loader, desc=f"ANN-TRAIN: Evaluating batches - XX/{data.num_samples / self.batch_size}"):
                input = input.to(device)
                #label = label.to(device)
                input = input.view(input.shape[0], -1)
                input.float()
                output = self(input)
                #torch.argmax(output, dim=1).cpu().numpy()
                #print(torch.argsort(output).cpu().numpy())
                result = torch.argsort(output).cpu().numpy()
                obj = []
                for rank in result:
                    ranking = pd.DataFrame({'ec': data.encoder.inverse_transform(rank), 'confidence': rank})

        #print(ranked_mappings)
        #ranking = pd.DataFrame({'ec':0, 'confidence': results})
        #ranking.sort_values(by="confidence", ascending=False, inplace=True, ignore_index=True)


class CustomDataset(Dataset):
    # Constructor
    def __init__(self, data):
        # Either read in the data from train.csv or just use it passed as-is
        self.encoder = preprocessing.LabelEncoder()
        self.encoder = self.encoder.fit(get_fndds_map())
        self.labels = self.encoder.fit_transform(data["ec"])
        self.data = torch.tensor(data.drop(columns=["ec", "upc"]).values, dtype=torch.float32)
        self.labels = torch.as_tensor(self.labels)
        self.num_samples = self.data.shape[0]
        print(f"ANN-DATA: {self.num_samples} samples were loaded")

    # Item access
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    # Length
    def __len__(self):
        return self.num_samples

    def num_cols(self):
        return self.data.shape[1]


def get_fndds_map(fndds_loc = "./data/raw/fndds_1718.csv"):
    fndds_data = pd.read_csv(fndds_loc, dtype=str, keep_default_na=False)
    food_codes = fndds_data["food_code"]
    food_codes = food_codes.append(pd.Series(["-99", "2053"]))
    return food_codes


def train(config):
    num_threads = config["general"]["num_threads"]
    # Read in the train set
    print("ANN-TRAIN: Loading the training set")
    train_data = pd.read_csv("./data/formatted/train.csv", dtype={"upc": str, "ec": str}, keep_default_na=False)
    input_size = len(train_data.columns)-2 # Magic -2 so we can drop the upc and ec columns
    output_size = len(get_fndds_map())
    print(f"ANN-TRAIN: {output_size} unique FNDDS classes identified for classification")

    # Prep the data
    print("ANN-TRAIN: Preparing the training set")
    train_set = CustomDataset(train_data)
    del train_data
    gc.collect()

    # Train the model
    #x = train_data.drop(columns="ec")
    #y = train_data["ec"]
    print("ANN-TRAIN: Training the model")
    model = FFNet([input_size, math.ceil((input_size - output_size) * .75), math.floor((input_size - output_size) * .25), output_size], config["method"]["model_loc"] + "ann/ann_model.pt")
    model.train_model(train_set)


def infer(config, data_loc="./data/formatted/train.csv", model_loc=""):
    #CHUNK_SIZE = 10000
    # Load the model
    print("ANN-INFER: Loading the model")
    model = (torch.load(config["method"]["model_loc"] + "ann/ann_model.pt") if not model_loc else torch.load(model_loc))

    # Read in the specified dataset
    print("ANN-INFER: Reading the provided dataset")
    test_data = pd.read_csv(data_loc, dtype={"upc": str, "ec": str}, keep_default_na=False)
    print("ANN-TRAIN: Preparing the provided set")
    test_set = CustomDataset(test_data)
    del test_data
    gc.collect()

    # Run the inference function
    model.infer_data(test_set)