# random_forest.py
# This utility implements the training and inference operations for our random forest method
# Auburn Big Data

# Internal Imports
import src.utilities.embedding.glove as glove
from src.utilities.official_eval.evaluation_script import evaluate
# External Imports
import time
import os
import json
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import pickle
import dill
import joblib
import sklearn.ensemble as skl
import sklearn.model_selection
import sklearn.metrics as skm
from src.utilities.user_types import ArgNamespace
from typing import Optional


def load_model(location = "./model/random_forest/rf_model.bin"):
    #rf = joblib.load(location)
    with open(location, "rb") as input_file:
        rf = dill.load(input_file)
        input_file.close()
    return rf


def train(config, parameters={}, train_loc="", save_name="", args: Optional[ArgNamespace] = None):
    args = None
    num_threads = config["general"]["num_threads"]
    # Read in the full training set
    print("RF-TRAIN: Loading the training data")
    train_data = pd.read_csv(("./data/formatted/train.csv" if not train_loc else train_loc), dtype={"upc":str, "ec":str}, keep_default_na=False)

    # Train the model
    x = train_data.drop(columns="ec")
    y = train_data["ec"]
    print("RF-TRAIN: Training the model")
    rf = skl.RandomForestClassifier(n_jobs=num_threads, bootstrap=False, verbose=2,
                                    n_estimators=(config["method"]["method_list"]["random_forest"][
                                                      "n_estimators"] if not "n_estimators" in parameters.keys() else
                                                  parameters["n_estimators"]),
                                    max_features=(config["method"]["method_list"]["random_forest"][
                                                      "max_features"] if not "max_features" in parameters.keys() else
                                                  parameters["max_features"]),
                                    max_depth=(config["method"]["method_list"]["random_forest"][
                                                   "max_depth"] if not "max_depth" in parameters.keys() else parameters[
                                        "max_depth"]),
                                    min_samples_split=(config["method"]["method_list"]["random_forest"][
                                                           "min_samples_split"] if not "min_samples_split" in parameters.keys() else
                                                       parameters["min_samples_split"]),
                                    min_samples_leaf=(config["method"]["method_list"]["random_forest"][
                                                          "min_samples_leaf"] if not "min_samples_leaf" in parameters.keys() else
                                                      parameters["min_samples_leaf"]))
    rf.fit(x, y)

    # Write the trained model to a file
    output_loc = config["method"]["model_loc"] + f"random_forest/{'rf_model.bin' if not save_name else save_name}"
    print(f"RF-TRAIN: Saving trained model to {output_loc}")
    #joblib.dump(rf, output_loc, compress=3)
    with open(output_loc, "wb") as output_file:
        dill.dump(rf, output_file, protocol=pickle.HIGHEST_PROTOCOL)


def infer(config, data_loc="", model_loc="", args: Optional[ArgNamespace] = None):
    args = None
    num_threads = config["general"]["num_threads"] * 2
    # If an alternate dataset was specified, split it into parts
    data_prefix = "./data/formatted/complete_data_test"
    if data_loc:
        print("RF-INFER: Splitting provided dataset for testing")
        # Get the data prefix ready
        data_prefix = ".".join(data_loc.split(".")[:-1])
        # Read in the data file
        print("RF-INFER: Reading provided dataset")
        data = pd.read_csv(data_loc, dtype={"upc":str, "ec":str}, keep_default_na=False)
        # Split the data into parts & write them
        counter = 1
        for item in np.array_split(data, num_threads):
            print(f"RF-INFER: Writing split {counter} to the filesystem")
            item.to_csv(f"{data_prefix}_{counter}.csv", index=False)
            counter += 1
        #del data

    # Load the model
    print("RF-INFER: Loading the random forest model")
    model = (load_model() if not model_loc else load_model(model_loc))

    # Split off into threads to process the results faster
    threads = []
    for counter in range(1, num_threads+1):
        inference_worker(counter, model, f"{data_prefix}_{counter}.csv", f"./data/random_forest/predictions_{counter}.csv")

        #t = mp.Process(target=inference_worker, args=(counter, model, f"{data_prefix}_{counter}.csv", f"./data/random_forest/predictions_{counter}.csv"))
        #threads.append(t)
        #t.start()
    #for thread in threads:
        #thread.join()

    # Merge the result files
    for counter in tqdm(range(1, num_threads+1), desc="RF-INFER: Joining the prediction rankings together..."):
        # Read in the file
        df = pd.read_csv(f"./data/random_forest/predictions_{counter}.csv", dtype=str, keep_default_na=False)
        if counter == 1:
            pd.DataFrame(columns=df.columns).to_csv("./data/random_forest/predictions.csv", index=False)
        df.to_csv("./data/random_forest/predictions.csv", mode="a", header=False, index=False)
    print("RF-INFER: Prediction rankings written successfully to the filesystem")

    # Remove the temporary prediction files
    for counter in range(1, num_threads+1):
        os.remove(f"{data_prefix}_{counter}.csv")
        os.remove(f"./data/random_forest/predictions_{counter}.csv")

    # Evaluate the results
    print("RF-EVAL: Reading prediction rankings & ground truth")
    truth_loc = "./data/formatted/test.csv" if not data_loc else data_loc
    ground_truth = pd.read_csv(truth_loc, dtype={"upc":str, "ec":str}, keep_default_na=False)
    ground_truth.drop(columns=ground_truth.columns[2:], inplace=True)
    #print(ground_truth)
    predictions = pd.read_csv("./data/random_forest/predictions.csv", dtype={"upc":str, "ec":str}, keep_default_na=False)
    #print(predictions)

    print("RF-EVAL: Evaluating the generated rankings")
    return evaluate(predictions, ground_truth)


def inference_worker(number, model, data_loc, save_loc, k=5):
    ppc_columns = ["upc", "ec", "confidence"]
    # Load in the data
    #print(f"RF-INFER-{number}: Loading testing data")
    data = pd.read_csv(data_loc, dtype={"upc": str, "ec": str}, keep_default_na=False)

    # Split the data into predictors and correct answers (x & y)
    x = data.drop(columns="ec")
    #y = data["ec"]

    # Loop over each item, recording the top-k items
    ppc_rankings = pd.DataFrame(columns=ppc_columns)
    model.verbose = 0
    for i, row in tqdm(x.iterrows(), desc=f"RF-TEST-{number}: Testing {len(data)} data points..."):
        # Initialize the object
        obj = []
        # Run inference & generate a ranking
        prediction = model.predict_proba([row.tolist()])[0]
        #ranking = pd.DataFrame({'upc':row["upc"], 'ec': model.classes_, 'confidence': prediction})
        ranking = pd.DataFrame({'ec': model.classes_, 'confidence': prediction})
        ranking.sort_values(by="confidence", ascending=False, inplace=True, ignore_index=True)
        #print(ranking)
        # Get the top-k items & add them to an object
        for i in range(0, k):
            obj.append({"upc":row["upc"], "ec":ranking["ec"][i], "confidence":ranking["confidence"][i]})
        # Append the object
        ppc_rankings = ppc_rankings.append(obj, ignore_index=True)

    # Write the rankings to a file
    #print(f"RF-INFER-{number}: Writing finished rankings to the filesystem")
    ppc_rankings.to_csv(save_loc, index=False)
    #print(f"RF-INFER-{number}: Rankings written to filesystem")


def hyperparameter_tune(config, args: Optional[ArgNamespace] = None):
    args = None
    hp_log = os.path.join(config['data']['path'], "random_forest/hyperparameter_log.txt")
    # Read the checkpoint file
    checkpoint = {}
    rf_hp_json = os.path.join(config['data']['path'], "temp/rf_hyperparameter.json")
    if os.path.exists(rf_hp_json):
        with open(rf_hp_json, "r") as checkpoint_file:
            checkpoint = json.load(checkpoint_file)
    else:
        # Overwrite the performance log
        with open(hp_log, "w") as logfile:
            logfile.write("Random Forest Hyperparameter Combination Performances:\n\n")
            logfile.close()
        # Initialize the checkpoint
        checkpoint = {
            "best_model": {"ndcg": 0.0, "success": 0.0},
            "best_parameters": {},
            "remaining_combinations": [],
        }
        # Build the grid
        grid = {
            "n_estimators": [int(x) for x in range(60, 85, 5)],
            "max_features": ["auto", "sqrt"],
            "max_depth": [int(x) for x in range(30, 50, 5)],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        # Construct a list of combinations to run
        keys, vals = zip(*grid.items())
        checkpoint["remaining_combinations"] = [dict(zip(keys, v)) for v in itertools.product(*vals)]

    # For each item in the grid, train a small-scale model & test it. Only keep a model if it performs higher than what is known
    while checkpoint["remaining_combinations"]:
        # Train the model
        train(config, parameters=checkpoint["remaining_combinations"][0], save_name="rf_hyperparameter.bin",
              train_loc="./data/formatted/hyperparameter_train.csv")
        # Evaluate the model
        ndcg, success = infer(config, data_loc="./data/formatted/hyperparameter_valid.csv",
                              model_loc=config["method"]["model_loc"] + "random_forest/rf_hyperparameter.bin")
        # Log performance
        with open(hp_log, "a") as logfile:
            logfile.write(f"Parameters: {json.dumps(checkpoint['remaining_combinations'][0])}\n\tNDCG@5:{ndcg}\n\tSuccess@5:{success}\n\n")

        # If the current model outperforms previous models, save it
        if (ndcg > checkpoint["best_model"]["ndcg"]) and (success > checkpoint["best_model"]["success"]):
            checkpoint["best_model"]["ndcg"] = ndcg
            checkpoint["best_model"]["success"] = success
            checkpoint["best_parameters"] = checkpoint["remaining_combinations"][0]
        # Remove the hyperparameter combination & write the checkpoint
        checkpoint["remaining_combinations"].pop(0)
        with open(rf_hp_json, "w") as checkpoint_file:
            json.dump(checkpoint, checkpoint_file)
