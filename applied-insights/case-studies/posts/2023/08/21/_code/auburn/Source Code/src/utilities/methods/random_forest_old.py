# random_forest.py
# This utility implements the training and inference operations for our random forest method
# Auburn Big Data

# Internal Imports
import src.utilities.embedding.glove as glove
from src.utilities.official_eval.evaluation_script import evaluate
# External Imports
import time
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import dill
import joblib
import sklearn.ensemble as skl
import sklearn.model_selection
import sklearn.metrics as skm


def load_model(location = "./model/random_forest/rf_model.bin"):
    #rf = joblib.load(location)
    with open(location, "rb") as input_file:
        rf = dill.load(input_file)
        input_file.close()
    return rf


def train(config):
    num_threads = config["general"]["num_threads"]
    # Start by reading in the IRI/PPC data
    print("RF-PREP: Reading in FNDDS/IRI data")
    #fndds_data = pd.read_csv("", dtype=str, keep_default_na=False)
    iri_data = pd.read_csv(f"./data/raw/iri_{config['database']['active_data']}.csv", dtype=str, keep_default_na=False)
    ppc_data = pd.read_csv(f"./data/raw/ppc_{config['database']['active_data']}.csv", dtype=str, keep_default_na=False)
    # Drop the index column
    iri_data.drop(columns=iri_data.columns[0], inplace=True)
    ppc_data.drop(columns=ppc_data.columns[0], inplace=True)
    # IRI: Drop the non-categorical columns, as well as any duplicates
    #iri_data.dropna(inplace=True)
    iri_data.drop_duplicates("upc", inplace=True)
    iri_data.drop(columns=iri_data.columns[1:10], inplace=True)
    # Convert all columns to integers
    for column in iri_data.columns:
        if column != "upc":
            iri_data[column] = pd.to_numeric(iri_data[column])
    for column in ppc_data.columns:
        if not (column == "upc" or column == "ec"):
            ppc_data[column] = pd.to_numeric(ppc_data[column])
    #print(iri_data.dtypes)
    #print(ppc_data.dtypes)

    ''''#tep = pd.read_csv("./data/embedded/iri_similarity_embedded_4.csv", dtype={"id":str}, keep_default_na=False)
    #iri_data.drop_duplicates("upc", inplace=True)
    iri_data.sort_values("upc", inplace=True, ignore_index=True)
    ppc_data.sort_values("upc", inplace=True, ignore_index=True)
    print(iri_data)
    print(ppc_data)
    #print(tep)
    test = ppc_data.merge(iri_data, left_on="upc", right_on="upc", how="inner").drop_duplicates("upc")

    print(test)
    #print(count)# 9137/9137, 9137/9137, 9137/9137, 9137/9137, ?, ?, ?, ?
    return'''

    # Get the average FNDDS embeddings (if needed)
    if (not os.path.exists("./data/embedded/fndds_combined_embedded.csv")):
        print("RF-PREP: Calculating average FNDDS embeddings")
        # Read in the FNDDS data
        fndds_data = pd.read_csv(f"./data/raw/fndds_{config['database']['active_data']}.csv", dtype=str, keep_default_na=False)
        fndds_data.drop(columns=fndds_data.columns[0], inplace=True)
        # Format the data as {"id":[], "desc":[]}
        fndds_data.rename(columns={"food_code":"id"}, inplace=True)
        fndds_data["desc"] = fndds_data["main_food_description"] + " " + fndds_data["additional_food_description"] + " " + fndds_data["ingredient_description"]
        fndds_data.drop(columns=["main_food_description", "additional_food_description", "ingredient_description", "wweia_category_description", "subcode_description"], inplace=True)
        # Get the average embeddings & save them to a file
        glove.get_average_embeddings(fndds_data, "./data/embedded/fndds_combined_embedded.csv", num_threads=num_threads)
    else:
        print("RF-PREP: Average FNDDS embeddings were already calculated. Using the previously-calculated data")

    # Read in the average FNDDS embeddings & store them
    fndds_data = pd.read_csv("./data/embedded/fndds_combined_embedded.csv", dtype=str, keep_default_na=False)
    fndds_data.drop(columns=fndds_data.columns[0:300], inplace=True)
    fndds_data.rename(columns={"combined_embed":"embed"}, inplace=True)
    fndds_data["embed"] = fndds_data["embed"].transform(lambda x: np.asarray(x[1:-1].split(', '), "float32"))
    #print(fndds_data)

    # Next, get the average embeddings for the IRI items, along with each item's similarity to every FNDDS item
    # This operation will take the longest, at roughly 5-6 hours, but will only be needed once
    if (not os.path.exists(f"./data/embedded/iri_similarity_embedded_{num_threads}.csv")):
        print("RF-PREP: Calculating average IRI embeddings and similarity scores")
        # Read in the IRI data
        temp_iri_data = pd.read_csv(f"./data/raw/iri_{config['database']['active_data']}.csv", dtype=str, keep_default_na=False)
        temp_iri_data.drop(columns=temp_iri_data.columns[0], inplace=True)
        # Format the data as {"id":[], "desc":[]}
        temp_iri_data.rename(columns={"upc":"id", "upcdesc":"desc"}, inplace=True)
        temp_iri_data.drop(columns=temp_iri_data.columns[2:], inplace=True)
        # Drop any null descriptions & duplicate values
        #print(len(temp_iri_data))
        #temp_iri_data.dropna(inplace=True)
        temp_iri_data.drop_duplicates("id", inplace=True)
        #print(len(temp_iri_data))#37936 -> corrected to 38133

        # Get the average similarities & save them to a file
        glove.get_average_similarities(temp_iri_data, fndds_data, "./data/embedded/iri_similarity_embedded.csv", num_threads=num_threads)
    else:
        print("RF-PREP: Average IRI embeddings and similarity scores were already calculated. Using the previously-calculated data")

    # Read in the IRI embeddings & similarities, storing them
    '''print("RF-PREP: Formatting data for training")
    temp_iri_data = pd.DataFrame()
    for chunk in tqdm(pd.read_csv("./data/embedded/iri_similarity_embedded.csv", chunksize=5000, dtype=str, keep_default_na=False), desc="Reading newly-formatted IRI data..."):
        temp_iri_data = temp_iri_data.append(chunk, ignore_index=True)
        #temp_iri_data = pd.concat([temp_iri_data, chunk], ignore_index=True)
    print(temp_iri_data)'''
    '''temp_iri_data = pd.read_csv("./data/embedded/iri_similarity_embedded.csv", chunksize=1000, dtype=str, keep_default_na=False)
    print("Data read. Merging the chunks now")
    temp_iri_data = pd.concat(temp_iri_data, ignore_index=True)
    print(temp_iri_data)'''

    #if (not (os.path.exists("./data/random_forest/train.csv") and os.path.exists("./data/random_forest/test.csv"))):
    if (not os.path.exists("./data/random_forest/train.csv")):
        # Read in the IRI embeddings & similarities, processing each one
        # Format the IRI categorical data
        combined_data = ppc_data.merge(iri_data, left_on="upc", right_on="upc", how="inner")
        combined_data.drop_duplicates("upc", inplace=True)
        #combined_data.dropna(inplace=True)
        combined_data.reset_index(inplace=True)
        combined_data.drop(columns=["index"], inplace=True)
        #print(combined_data)
        #print(len(combined_data))
        # Format the IRI data properly as: [ID, categorical variables, desc, cos_sim, euc_sim]
        splits = {"train":config["method"]["method_list"]["random_forest"]["train_size"], "test":config["method"]["method_list"]["random_forest"]["test_size"]}
        readfile_split = "./data/embedded/iri_similarity_embedded.csv".split(".")
        readfile_split = [".".join(readfile_split[:-1]), readfile_split[-1]]
        writefile_split = "./data/random_forest/complete_data.csv".split(".")
        writefile_split = [".".join(writefile_split[:-1]), writefile_split[-1]]
        threads = []
        num = 1
        #rf_format_worker(num, combined_data, splits, f"{readfile_split[0]}_{num}.{readfile_split[1]}", {"train":f"{writefile_split[0]}_train_{num}.{writefile_split[1]}", "test":f"{writefile_split[0]}_test_{num}.{writefile_split[1]}"})
        for counter in range(1, num_threads+1):
            t = mp.Process(target=rf_format_worker, args=(counter, combined_data, splits, f"{readfile_split[0]}_{counter}.{readfile_split[1]}", {"train":f"{writefile_split[0]}_train_{counter}.{writefile_split[1]}", "test":f"{writefile_split[0]}_test_{counter}.{writefile_split[1]}"}))
            threads.append(t)
            t.start()
        for thread in threads:
            thread.join()

        # Combine all the training data into one set
        for counter in tqdm(range(1, num_threads+1), desc="RF-FORMAT: Joining the training/testing sets together..."):
            # Process the training data first (and initialize the full set file)
            df = []
            train_init=True
            # NOTE: file_exists IS NOT A FUNCTION, replace this with a function that determines whether the file exists or not
            if (file_exists(f"{writefile_split[0]}_train_{counter}.{writefile_split[1]}")):
                df = pd.read_csv(f"{writefile_split[0]}_train_{counter}.{writefile_split[1]}", dtype={"upc": str, "ec": str}, keep_default_na=False)
                if counter == 1:
                    pd.DataFrame(columns=df.columns).to_csv("./data/random_forest/train.csv", index=False)
                    pd.DataFrame(columns=df.columns).to_csv("./data/random_forest/full_set.csv", index=False)
                df.to_csv("./data/random_forest/train.csv", mode="a", header=False, index=False)
                df.to_csv("./data/random_forest/full_set.csv", mode="a", header=False, index=False)
            else:
                train_init=False
            # Next, process the testing data
            if (file_exists(f"{writefile_split[0]}_test_{counter}.{writefile_split[1]}")):
                df = pd.read_csv(f"{writefile_split[0]}_test_{counter}.{writefile_split[1]}", dtype={"upc":str, "ec":str}, keep_default_na=False)
                if counter == 1:
                    pd.DataFrame(columns=df.columns).to_csv("./data/random_forest/test.csv", index=False)
                    if not train_init:
                        pd.DataFrame(columns=df.columns).to_csv("./data/random_forest/full_set.csv", index=False)
                df.to_csv("./data/random_forest/test.csv", mode="a", header=False, index=False)
                df.to_csv("./data/random_forest/full_set.csv", mode="a", header=False, index=False)
    else:
        print("RF-FORMAT: Train/test splits were already created. Using the previously-created splits")

    # If the config file only wanted the data prepared, return here
    if (config["method"]["method_list"]["random_forest"]["only_prep_data"]):
        print("RF-PREP: Data preparation finished and configuration indicates not to train a model")
        return

    # Read in the full training set
    print("RF-TRAIN: Loading the training data")
    train_data = pd.read_csv("./data/random_forest/train.csv", dtype={"upc":str, "ec":str}, keep_default_na=False)

    # Split the data into train/test splits & write these to disk
    #splits = sklearn.model_selection.train_test_split(combined_data, train_size=.6, test_size=.4)
    #splits[0].to_csv(f"{config['data']['path']}random_forest/train.csv")
    #splits[1].to_csv(f"{config['data']['path']}random_forest/test.csv")

    # Train the model
    x = train_data.drop(columns="ec")
    y = train_data["ec"]
    print("RF-TRAIN: Training the model")
    rf = skl.RandomForestClassifier(n_jobs=num_threads, bootstrap=False, max_depth=35, n_estimators=75, verbose=5)
    rf.fit(x, y)

    # Write the trained model to a file
    output_loc = config["method"]["model_loc"] + "random_forest/rf_model.bin"
    print(f"RF-TRAIN: Saving trained model to {output_loc}")
    #joblib.dump(rf, output_loc, compress=3)
    with open(output_loc, "wb") as output_file:
        dill.dump(rf, output_file)
        output_file.close()


def rf_format_worker(number, ppc_cat_data, splits, read_loc, save_loc):
    print(f"RF-FORMAT-{number}: Reading IRI data from file")
    # Read in the IRI data
    iri_data = pd.read_csv(read_loc, dtype={"id":str}, keep_default_na=False)
    #print(iri_data)

    print(f"RF-FORMAT-{number}: Formatting full dataset")
    # Combine the IRI data with the PPC data
    result_data = ppc_cat_data.merge(iri_data, left_on="upc", right_on="id", how="inner")
    #print(result_data)
    #print(pd.concat([result_data["upc"], result_data["id"], result_data["ec"]], axis=1))
    result_data.drop(columns=["id"], inplace=True)
    # Combine the running dataset with the categorical IRI variables
    #result_data = result_data.merge(iri_cat_data, left_on="upc", right_on="upc", how="inner", copy=False)
    #result_data.drop_duplicates("ec", inplace=True)
    result_data.dropna(inplace=True)
    #print(len(result_data))
    #print(result_data)

    print(f"RF-FORMAT-{number}: Splitting data ({len(result_data)} items) into train/test sets")
    if (splits["test"] == 1):
        data_splits = {"test":result_data}
    elif (splits["train"] == 1):
        data_splits = {"train":result_data}
    else:
        data_splits = sklearn.model_selection.train_test_split(result_data, train_size=splits["train"], test_size=splits["test"])

    print(f"RF-FORMAT-{number}: Writing formatted data to filesystem")
    keys = data_splits.keys()
    if ("train" in keys):
        data_splits[0].to_csv(save_loc["train"], index=False)
        print(f"RF-FORMAT-{number}: Training data written to filesystem")
    if ("test" in keys):
        data_splits[1].to_csv(save_loc["test"], index=False)
        print(f"RF-FORMAT-{number}: Testing data written to filesystem")


def test(config, location = ""):
    # Load the model
    print("RF-TEST: Loading the random forest model")
    model = (load_model() if not location else load_model(location))

    # Read in the testing data
    print("RF-TEST: Loading the testing data")
    data = pd.read_csv("./data/random_forest/test.csv", dtype={"upc":str, "ec":str}, keep_default_na=False)
    #data.drop(columns=data.columns[0], inplace=True)
    #for column in data.columns:
        #data[column] = pd.to_numeric(data[column])
    #print(data)

    # Split the data into predictors and correct answers (x & y)
    x = data.drop(columns="ec")
    y = data["ec"]

    # Get the explained variance
    print("RF-TEST: Calculating explained variance")
    exp_var = skm.explained_variance_score(y, x)
    print(f"RF-TEST: Explained variance is {exp_var}")

    # For each item in the data, generate a ranking
    '''print(f"RF-TEST: Testing {len(data)} data points")
    num_correct = 0
    model.verbose = 0
    for i in tqdm(range(len(data))):
        # Run inference & generate a ranking
        prediction = model.predict_proba(x.iloc[[i]])[0]
        ranking = pd.DataFrame({'ec':model.classes_, 'confidence':prediction})
        ranking.sort_values(by="confidence", ascending=False, inplace=True)
        #print(ranking)
        for i in range(0, 5):
            if (int(ranking.iloc[i]["ec"]) == int(y.iloc[i])):
                num_correct += 1

        # Using the ranking, grab the top-k entries and save them to a file
    # Log the number correct on the testing set
    print(f"RF-TEST: Got {num_correct}/{len(data)} correct (Estimated performance: {num_correct/len(data)})")'''


def infer_singlethread(config, dataloc="", model_loc=""):
    # Load the model
    print("RF-INFER: Loading the random forest model")
    model = (load_model() if not model_loc else load_model(model_loc))

    ppc_columns = ["upc", "ec", "confidence"]
    # Load in the data
    print("RF-INFER: Loading testing data")
    data_loc = "./data/random_forest/test.csv" if not dataloc else dataloc
    data = pd.read_csv(data_loc, dtype={"upc": str, "ec": str}, keep_default_na=False)

    # Split the data into predictors and correct answers (x & y)
    x = data.drop(columns="ec")
    #y = data["ec"]

    # Loop over each item, recording the top-k items
    ppc_rankings = pd.DataFrame(columns=ppc_columns)
    model.verbose = 0
    for i, row in tqdm(x.iterrows(), desc=f"RF-TEST: Testing {len(data)} data points..."):
        # Initialize the object
        obj = []
        # Run inference & generate a ranking
        prediction = model.predict_proba([row.tolist()])[0]
        # ranking = pd.DataFrame({'upc':row["upc"], 'ec': model.classes_, 'confidence': prediction})
        ranking = pd.DataFrame({'ec': model.classes_, 'confidence': prediction})
        ranking.sort_values(by="confidence", ascending=False, inplace=True, ignore_index=True)
        # print(ranking)
        # Get the top-k items & add them to an object
        for i in range(0, k):
            obj.append({"upc": row["upc"], "ec": ranking["ec"][i], "confidence": ranking["confidence"][i]})
        # Append the object
        ppc_rankings = ppc_rankings.append(obj, ignore_index=True)

    # Write the rankings to a file
    print(f"RF-INFER: Writing finished rankings to the filesystem")
    ppc_rankings.to_csv("./data/random_forest/predictions.csv", index=False)
    print(f"RF-INFER: Rankings written to filesystem")

    # Evaluate the results
    print("RF-EVAL: Reading prediction rankings & ground truth")
    ground_truth = pd.read_csv(data_loc, dtype={"upc": str, "ec": str}, keep_default_na=False)
    ground_truth.drop(columns=ground_truth.columns[2:], inplace=True)
    # print(ground_truth)
    predictions = pd.read_csv("./data/random_forest/predictions.csv", dtype={"upc": str, "ec": str},
                              keep_default_na=False)
    # print(predictions)

    print("RF-EVAL: Evaluating the generated rankings")
    evaluate(predictions, ground_truth)


def infer(config, data_loc="", model_loc=""):
    num_threads = config["general"]["num_threads"] * 2
    # If an alternate dataset was specified, split it into parts
    data_prefix = "./data/random_forest/complete_data_test"
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
        os.remove(f"./data/random_forest/predictions_{counter}.csv")

    # Evaluate the results
    print("RF-EVAL: Reading prediction rankings & ground truth")
    truth_loc = "./data/random_forest/test.csv" if not data_loc else data_loc
    ground_truth = pd.read_csv(truth_loc, dtype={"upc":str, "ec":str}, keep_default_na=False)
    ground_truth.drop(columns=ground_truth.columns[2:], inplace=True)
    #print(ground_truth)
    predictions = pd.read_csv("./data/random_forest/predictions.csv", dtype={"upc":str, "ec":str}, keep_default_na=False)
    #print(predictions)

    print("RF-EVAL: Evaluating the generated rankings")
    evaluate(predictions, ground_truth)


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