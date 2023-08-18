# bert.py
# This utility allows us to easily use BERT-based contextualized embeddings
# Auburn Big Data
# NOTE: for future, we want to assess the differences between a fine-tuned BERT and a non-fine-tuned BERT (starting with non-fine-tuned)

# Internal Imports
# External Imports
import torch
from transformers import BertTokenizer, BertModel
import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import distance


def load_model(file_loc = "P:/pr-westat-food-for-thought/models/bert/bert-base-uncased"):
    model = BertModel.from_pretrained(file_loc)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(file_loc)
    return model, tokenizer


def get_avg_text_vector(text, model, tokenizer):
    output = [0] * 768
    if len(text):
        encoding = tokenizer.encode_plus(text, add_special_tokens=False, return_attention_mask=True)
        input = torch.tensor(encoding.get('input_ids')).unsqueeze(0).to(torch.int64)
        mask = torch.tensor(encoding.get('attention_mask')).unsqueeze(0)
        with torch.no_grad():
            output = model(input, mask)[0]# This line throws an error on the forward call accessing index 0 of the hidden states (Iteration 164 on thread 1)
            # Remove the batches (dim 0)
            output = torch.squeeze(output, dim=0)
            # Get the average embedding
            output = torch.mean(output, dim=0).tolist()
    return output


def get_avg_embeds(data, save_loc, num_threads=8, model_loc=""):
    tmp_dir = "./data/temp/"
    # Load the model & tokenizer
    model, tokenizer = (load_model() if not model_loc else load_model(model_loc))

    # Split the data into chunks
    counter = 1
    for x in np.array_split(data, num_threads, axis=0):
        x.to_csv(f"{tmp_dir}datasplit_{counter}.csv")
        counter += 1

    # Launch threads to process the data
    threads = []
    for counter in range(1, num_threads + 1):
        t = mp.Process(target=avg_embed_worker, args=(counter, model, tokenizer, f"{tmp_dir}datasplit_{counter}.csv", f"{tmp_dir}resultsplit_{counter}.csv"))
        threads.append(t)
        t.start()

    # Join the threads back
    for thread in threads:
        thread.join()

    # Combine the result files and write them to the save location
    for counter in range(1, num_threads+1):
        df = pd.read_csv(f"{tmp_dir}resultsplit_{counter}.csv", dtype=str, keep_default_na=False)
        if counter == 1:
            pd.DataFrame(columns=df.columns).to_csv(save_loc, index=False)
        df.to_csv(save_loc, mode="a", header=False, index=False)

    # Re-read the data and reformat it for later use
    data = pd.read_csv(save_loc, dtype=str, keep_default_na=False)
    # Break the array column into multiple columns
    new_data = data["desc"].transform(lambda x: np.asarray(x[1:-1].split(', '), "float32")).apply(pd.Series)
    # Rename the new columns & add in the ID & combined column again
    result_data = new_data.rename(lambda x: f"embed_{x}", axis=1)
    result_data["id"] = data["id"]
    result_data["combined_embed"] = data["desc"]
    # Write the data to teh file again
    result_data.to_csv(save_loc, index=False)

    # Remove all temporary files
    for counter in range(1, num_threads+1):
        os.remove(f"{tmp_dir}datasplit_{counter}.csv")
        os.remove(f"{tmp_dir}resultsplit_{counter}.csv")


def avg_embed_worker(number, model, tokenizer, data_loc, save_loc):
    # Read the data file
    df = pd.read_csv(data_loc, dtype=str, keep_default_na=False)
    df.drop(columns=df.columns[0], inplace=True)

    # Write the initial header
    pd.DataFrame(columns=df.columns[0:2]).to_csv(save_loc, index=False)

    # For each entry in the file, get the average embedding
    temp_embed = []
    for i, row in tqdm(df.iterrows(), desc=f"THREAD-{number}: Calculating averaged BERT embeddings..."):
        # Append the averaged text vector
        temp_embed.append({
            "id":row["id"],
            "desc":get_avg_text_vector(row["desc"], model, tokenizer)
        })

    # Write the result to a file
    pd.DataFrame(temp_embed, columns=df.columns[0:2]).to_csv(save_loc, mode="a", header=False, index=False)


def get_avg_sim(data, reference, save_loc, num_threads=0, model_loc=""):
    tmp_dir = "./data/temp/"
    # Prepare by loading in the embedding
    model, tokenizer = (load_model() if not model_loc else load_model(model_loc))

    # Now, split the data into chunks to be processed in parallel
    counter = 1
    for x in np.array_split(data, num_threads, axis=0):
        x.to_csv(f"{tmp_dir}datasplit_{counter}.csv")
        counter += 1

    # Launch threads to process each file
    threads = []
    for counter in range(1, num_threads + 1):
        t = mp.Process(target=avg_sim_worker, args=(counter, model, tokenizer, reference, f"{tmp_dir}datasplit_{counter}.csv", f"{tmp_dir}resultsplit_{counter}.csv"))
        threads.append(t)
        t.start()

    # Join the threads back
    for thread in threads:
        thread.join()

    # Read in each result file, reformat it, and save it
    filename_split = save_loc.split(".")
    filename_split = [".".join(filename_split[:-1]), filename_split[-1]]
    threads = []
    print("BERT-SIM: Formatting embeddings and similarities...")
    for counter in range(1, num_threads + 1):
        t = mp.Process(target=sim_format_worker, args=(counter, f"{tmp_dir}resultsplit_{counter}.csv", f"{filename_split[0]}_{counter}.{filename_split[1]}"))
        threads.append(t)
        t.start()
    for thread in threads:
        thread.join()

    # Combine the result files & write them to the save location
    for counter in tqdm(range(1, num_threads + 1), desc="BERT-SIM: Combining result files..."):
        df = pd.read_csv(f"{filename_split[0]}_{counter}.{filename_split[1]}", dtype=str, keep_default_na=False)
        if counter == 1:
            pd.DataFrame(columns=df.columns).to_csv(save_loc, index=False)
        df.to_csv(save_loc, mode="a", header=False, index=False)

    # Remove all temporary files
    for counter in range(1, num_threads + 1):
        os.remove(f"{tmp_dir}datasplit_{counter}.csv")
        os.remove(f"{tmp_dir}resultsplit_{counter}.csv")


def avg_sim_worker(number, model, tokenizer, reference, data_loc, save_loc):
    # Read the data file
    df = pd.read_csv(data_loc, dtype=str, keep_default_na=False)
    df.drop(columns=df.columns[0], inplace=True)

    # Write the initial header to the file
    columns = np.append(np.asarray(df.columns[0:2]), ["cos_sim", "euc_sim"])
    pd.DataFrame(columns=columns).to_csv(save_loc, index=False)

    # For each entry in the file, get the average embedding, cosine similarity, and euclidean similarity
    temp_embed = []
    for i, row in tqdm(df.iterrows(), desc=f"THREAD-{number}: Calculating averaged BERT embeddings..."):
        # Calculate the embedding
        embed = get_avg_text_vector(row["desc"], model, tokenizer)
        # Append the averaged text vector
        temp_embed.append({
            "id": row["id"],
            "desc": embed,
            "cos_sim": reference["embed"].transform(lambda x: distance.cosine(embed, x)).tolist(),
            "euc_sim": reference["embed"].transform(lambda x: distance.euclidean(embed, x)).tolist()
        })

    # Write the result to the file
    pd.DataFrame(temp_embed, columns=columns).to_csv(save_loc, mode="a", header=False, index=False)


def sim_format_worker(number, data_loc, save_loc):
    # Read in the data
    data = pd.read_csv(data_loc, dtype=str, keep_default_na=False)

    # Reformat the dataframe
    # Break the embedded description into multiple columns
    new_data = data["desc"].transform(lambda x: np.asarray(x[1:-1].split(', '), "float32")).apply(pd.Series)
    # Rename the new columns and create the resultant dataframe
    result_data = pd.concat([data["id"], new_data.rename(lambda x: f"embed_{x}", axis=1)], axis=1)
    # Break the cosine similarity column into multiple columns
    new_data = data["cos_sim"].transform(lambda x: np.asarray(x[1:-1].split(', '), "float32")).apply(pd.Series)
    result_data = pd.concat([result_data, new_data.rename(lambda x: f"cos_sim_{x}", axis=1)], axis=1)
    # Break the euclidean similarity into multiple columns
    new_data = data["euc_sim"].transform(lambda x: np.asarray(x[1:-1].split(', '), "float32")).apply(pd.Series)
    result_data = pd.concat([result_data, new_data.rename(lambda x: f"euc_sim_{x}", axis=1)], axis=1)

    print(f"BERT-SIM: File {number} has finished processing. Writing it to disk")
    # Write the data to a file
    result_data.to_csv(save_loc, index=False)
    print(f"BERT-SIM: File {number} has been written")


if (__name__=="__main__"):
    print("Testing BERT compatibility")
    model, tokenizer = load_model()
    # Embed the following sentences
    sentences = [
        "This is a test",
        "glutenous wholesale bread, on a bun",
        "I really need to think of some good examples here",
        "I wonder what approaches the other teams are using for this competition"
    ]
    for line in sentences:
        print(get_avg_text_vector(line, model, tokenizer))