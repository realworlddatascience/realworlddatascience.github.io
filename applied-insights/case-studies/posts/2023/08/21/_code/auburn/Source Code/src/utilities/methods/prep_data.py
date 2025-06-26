# prep_data.py
# This utility manages formatting the FNDDS and IRI data properly for the downstream models/methods
# Auburn Big Data

# Internal Imports
import src.interface.embedding as embedding
# External Imports
import os
import pandas as pd
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from sklearn import model_selection
# import sklearn.model_selection


def prep_data(config):
    num_threads = config["general"]["num_threads"]

    # Prep the FNDDS data, then read it in
    prep_fndds(config, num_threads)
    fndds_data = pd.read_csv("./data/embedded/fndds_combined_embedded.csv", dtype=str, keep_default_na=False)
    fndds_data.drop(columns=fndds_data.columns[0:300], inplace=True)
    fndds_data.rename(columns={"combined_embed": "embed"}, inplace=True)
    fndds_data["embed"] = fndds_data["embed"].transform(lambda x: np.asarray(x[1:-1].split(', '), "float32"))

    # Next, get the average embeddings for the IRI items, along with each item's similarity to every given FNDDs item
    # This operation will take the longest, on the order of multiple hours, but will only be needed once
    prep_iri(config, num_threads, fndds_data)

    # Read in the IRI/PPC data
    print("DATA-PREP: Reading in FNDDS/IRI data")
    iri_data = pd.read_csv(f"./data/raw/iri_{config['database']['active_data']}.csv", dtype=str, keep_default_na=False)
    ppc_data = pd.read_csv(f"./data/raw/ppc_{config['database']['active_data']}.csv", dtype=str, keep_default_na=False)
    # Drop the index column
    iri_data.drop(columns=iri_data.columns[0], inplace=True)
    ppc_data.drop(columns=ppc_data.columns[0], inplace=True)
    # IRI: Drop the non-categorical columns, as well as any duplicates
    iri_data.drop_duplicates("upc", inplace=True)
    iri_data.drop(columns=iri_data.columns[1:10], inplace=True)
    # Convert all columns to integers
    for column in iri_data.columns:
        if column != "upc":
            iri_data[column] = pd.to_numeric(iri_data[column])
    for column in ppc_data.columns:
        if not (column == "upc" or column == "ec"):
            ppc_data[column] = pd.to_numeric(ppc_data[column])

    # Now, split the data
    split_data(config, num_threads, iri_data, ppc_data)


def prep_fndds(config, num_threads):
    if not os.path.exists("./data/embedded/fndds_combined_embedded.csv"):
        print("DATA-PREP: Calculating average FNDDS embeddings")
        # Read in the FNDDS data
        fndds_data = pd.read_csv(f"./data/raw/fndds_{config['database']['active_data']}.csv", dtype=str, keep_default_na=False)
        fndds_data.drop(columns=fndds_data.columns[0], inplace=True)
        # Format the data as {"id":[], "desc":[]}
        fndds_data.rename(columns={"food_code": "id"}, inplace=True)
        fndds_data["desc"] = fndds_data["main_food_description"] + " " + fndds_data["additional_food_description"] + " " + fndds_data["ingredient_description"]
        fndds_data.drop(columns=["main_food_description", "additional_food_description", "ingredient_description", "wweia_category_description", "subcode_description"], inplace=True)
        # Get the average embeddings & save them to a file
        embedding.get_avg_embeds(config["embedding"]["active_embedding"], fndds_data, "./data/embedded/fndds_combined_embedded.csv", num_threads=num_threads)
    else:
        print("DATA-PREP: Average FNDDS embeddings were already calculated. Using the previously-calculated data")


def prep_iri(config, num_threads, fndds_data):
    if not os.path.exists(f"./data/embedded/iri_similarity_embedded.csv"):#iri_similarity_embedded_{num_threads}.csv
        print("DATA-PREP: Calculating average IRI embeddings and similarity scores")
        # Read in the IRI data
        iri_data = pd.read_csv(f"./data/raw/iri_{config['database']['active_data']}.csv", dtype=str, keep_default_na=False)
        iri_data.drop(columns=iri_data.columns[0], inplace=True)
        # Format the data as {"id":[], "desc":[]}
        iri_data.rename(columns={"upc": "id", "upcdesc": "desc"}, inplace=True)
        iri_data.drop(columns=iri_data.columns[2:], inplace=True)
        # Drop any duplicate values
        iri_data.drop_duplicates("id", inplace=True)

        # Get the average similarities & save them to a file
        embedding.get_avg_sim(config["embedding"]["active_embedding"], iri_data, fndds_data, "./data/embedded/iri_similarity_embedded.csv", num_threads=num_threads)
    else:
        print("DATA-PREP: Average IRI embeddings and similarity scores were already calculated. Using the previously-calculated data")


def split_data(config, num_threads, iri_data, ppc_data):
    if (not os.path.exists("./data/formatted/train.csv")) and (not os.path.exists("./data/formatted/test.csv")):
        print(f"DATA-FORMAT: Splitting data into train/test splits ({config['method']['train_size']} train/{config['method']['test_size']} test)")
        # Read in the IRI embeddings and similarities, processing each one
        # Format the IRI categorical data
        combined_data = ppc_data.merge(iri_data, left_on="upc", right_on="upc", how="inner")
        combined_data.drop_duplicates("upc", inplace=True)
        combined_data.reset_index(inplace=True)
        combined_data.drop(columns=["index"], inplace=True)
        # Format the IRI data properly as: [ID, categorical variables, desc, cos_sim, euc_sim]
        splits = {"train": config["method"]["train_size"], "test": config["method"]["test_size"]}
        readfile_split = "./data/embedded/iri_similarity_embedded.csv".split(".")
        readfile_split = [".".join(readfile_split[:-1]), readfile_split[-1]]
        writefile_split = "./data/formatted/complete_data.csv".split(".")
        writefile_split = [".".join(writefile_split[:-1]), writefile_split[-1]]
        threads = []
        num = 1
        for counter in range(1, num_threads + 1):
            t = mp.Process(target=data_format_worker, args=(counter, combined_data, splits, f"{readfile_split[0]}_{counter}.{readfile_split[1]}",
            {"train": f"{writefile_split[0]}_train_{counter}.{writefile_split[1]}",
             "test": f"{writefile_split[0]}_test_{counter}.{writefile_split[1]}"}))
            threads.append(t)
            t.start()
        for thread in threads:
            thread.join()

        # Combine all the training/testing data into one set
        for counter in tqdm(range(1, num_threads + 1), desc="DATA-FORMAT: Joining the training/testing sets together..."):
            # Process the training data first (and initialize the full set file)
            df = []
            train_init=True
            if os.path.exists(f"{writefile_split[0]}_train_{counter}.{writefile_split[1]}"):
                df = pd.read_csv(f"{writefile_split[0]}_train_{counter}.{writefile_split[1]}", dtype={"upc": str, "ec": str}, keep_default_na=False)
                if counter == 1:
                    pd.DataFrame(columns=df.columns).to_csv("./data/formatted/train.csv", index=False)
                    pd.DataFrame(columns=df.columns).to_csv("./data/formatted/full_set.csv", index=False)
                df.to_csv("./data/formatted/train.csv", mode="a", header=False, index=False)
                df.to_csv("./data/formatted/full_set.csv", mode="a", header=False, index=False)
            else:
                train_init=False
            # Next, process the testing data
            if os.path.exists(f"{writefile_split[0]}_test_{counter}.{writefile_split[1]}"):
                df = pd.read_csv(f"{writefile_split[0]}_test_{counter}.{writefile_split[1]}", dtype={"upc": str, "ec": str}, keep_default_na=False)
                if counter == 1:
                    pd.DataFrame(columns=df.columns).to_csv("./data/formatted/test.csv", index=False)
                    if not train_init:
                        pd.DataFrame(columns=df.columns).to_csv("./data/formatted/full_set.csv", index=False)
                df.to_csv("./data/formatted/test.csv", mode="a", header=False, index=False)
                df.to_csv("./data/formatted/full_set.csv", mode="a", header=False, index=False)
    else:
        print("DATA-FORMAT: Train/test splits were already created. Using the previously-created splits")


def data_format_worker(number, ppc_cat_data, splits, read_loc, save_loc):
    print(f"DATA-FORMAT-{number}: Step 1/4 - Reading IRI data from file")
    # Read in the IRI data
    iri_data = pd.read_csv(read_loc, dtype={"id": str}, keep_default_na=False)

    print(f"DATA-FORMAT-{number}: Step 2/4 - Formatting full dataset")
    # Combine the IRI data with the PPC data
    result_data = ppc_cat_data.merge(iri_data, left_on="upc", right_on="id", how="inner")
    result_data.drop(columns=["id"], inplace=True)
    # Drop any null values from the running dataset
    result_data.dropna(inplace=True)

    print(f"DATA-FORMAT-{number}: Step 3/4 - Splitting data ({len(result_data)} items) into train/test sets")
    keys = []
    if (splits["test"] == 1):
        data_splits = {"test": result_data}
        keys = ["test"]
    elif (splits["train"] == 1):
        data_splits = {"train": result_data}
        keys = ["train"]
    else:
        data_splits = model_selection.train_test_split(result_data, train_size=splits["train"], test_size=splits["test"], random_state=3250)
        data_splits = {'train': data_splits[0], 'test': data_splits[1]}
        keys = ["train", "test"]

    print(f"DATA-FORMAT-{number}: Step 4/4 - Writing formatted data to filesystem")
    if ("train" in keys):
        data_splits['train'].to_csv(save_loc["train"], index=False)
        print(f"DATA-FORMAT-{number}: Training data written to filesystem")
    if ("test" in keys):
        data_splits['test'].to_csv(save_loc["test"], index=False)
        print(f"DATA-FORMAT-{number}: Testing data written to filesystem")


# def split_data_tuning(config, train_loc: str = ""):
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