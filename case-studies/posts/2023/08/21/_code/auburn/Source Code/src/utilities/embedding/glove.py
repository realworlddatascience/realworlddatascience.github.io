# glove.py
# This utility allows us to easily use GloVe embeddings
# Auburn Big Data

import gc
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from scipy.spatial import distance
from tqdm import tqdm


def load(file_loc = "./embedding/glove/glove.6B.300d.txt"):
    f = open(file_loc, encoding="utf8")
    embedding = {}
    for i, line in enumerate(tqdm(f, desc="GLOVE: Reading embeddings...")):
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embedding[word] = vector
    return embedding


def get_avg_text_vector(text, embedding):
    sum_of_vector = np.full((len(embedding[list(embedding.keys())[0]]), ), 0)
    word_vec = text.split()
    counter = 0
    for word in word_vec:
        word_embedding = embedding.get(word, np.full((len(embedding[list(embedding.keys())[0]]), ), 0))
        if np.all(word_embedding == 0):
            counter += 1
        sum_of_vector = np.add(word_embedding, sum_of_vector)
    vector_len = len(word_vec) - counter
    if vector_len <= 0:
        vector_len = 1
    sum_of_vector = np.divide(sum_of_vector, vector_len)
    return np.ndarray.tolist(sum_of_vector)


def get_average_embeddings(data, save_loc, num_threads=8):
    tmp_dir = "./data/temp/"
    # Prepare by loading in the embedding
    embed = load()

    # Now, split the data into chunks to be processed in parallel
    counter = 1
    for x in np.array_split(data, num_threads, axis=0):
        x.to_csv(f"{tmp_dir}datasplit_{counter}.csv")
        counter += 1
    del data
    gc.collect()

    # Launch threads to process each file
    threads = []
    for counter in range(1, num_threads+1):
        t = mp.Process(target=average_embedding_worker, args=(counter, embed, f"{tmp_dir}datasplit_{counter}.csv", f"{tmp_dir}resultsplit_{counter}.csv"))
        threads.append(t)
        t.start()

    # Join the threads back
    for thread in threads:
        thread.join()

    # Combine the result files & write them to the save location
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
    # Write the data to the file again
    result_data.to_csv(save_loc, index=False)

    # Remove all temporary files
    for counter in range(1, num_threads+1):
        os.remove(f"{tmp_dir}datasplit_{counter}.csv")
        os.remove(f"{tmp_dir}resultsplit_{counter}.csv")


def batched_average_embedding_worker(number, embedding, data_loc, save_loc):
    CHUNK_SIZE = 2500 # Add this to config later
    # For each chunk, do this processing
    prev_len = 2500
    chunk_counter = 1
    while prev_len == CHUNK_SIZE:
        # Read the csv
        if chunk_counter == 1:
            df = pd.read_csv(data_loc, dtype=str, keep_default_na=False, nrows=CHUNK_SIZE)
        else:
            df = pd.read_csv(data_loc, dtype=str, keep_default_na=False, nrows=CHUNK_SIZE, skiprows=(CHUNK_SIZE*(chunk_counter-1)))
        df.drop(columns=df.columns[0], inplace=True)

        # If this is the first loop, write the initial header
        if chunk_counter == 1:
            pd.DataFrame(columns=df.columns[0:2]).to_csv(save_loc, index=False)
        chunk_counter += 1

        # For each entry, get the average embedding
        temp_embed = []
        for i, row in tqdm(df.iterrows(), desc=f"THREAD-{number}: Calculating averaged GloVe embeddings... (Chunk {chunk_counter})"):
            # Append the averaged text vector
            temp_embed.append({
                "id": row["id"],
                "desc": get_avg_text_vector(row["desc"], embedding)
            })

        # Write the result to the file
        pd.DataFrame(temp_embed, columns=df.columns[0:2]).to_csv(save_loc, mode="a", header=False, index=False)



def average_embedding_worker(number, embedding, data_loc, save_loc):
    # Read the data file
    df = pd.read_csv(data_loc, dtype=str, keep_default_na=False)
    df.drop(columns=df.columns[0], inplace=True)

    # Write the initial header to the file
    pd.DataFrame(columns=df.columns[0:2]).to_csv(save_loc, index=False)

    # For each entry in the file, get the average embedding
    temp_embed = []
    for i, row in tqdm(df.iterrows(), desc=f"THREAD-{number}: Calculating averaged GloVe embeddings..."):
        # Append the averaged text vector
        temp_embed.append({
            "id":row["id"],
            "desc":get_avg_text_vector(row["desc"], embedding)
        })

    # Write the result to the file
    pd.DataFrame(temp_embed, columns=df.columns[0:2]).to_csv(save_loc, mode="a", header=False, index=False)


def get_average_similarities(data, ref_data, save_loc, num_threads=8):
    tmp_dir = "./data/temp/"
    # Prepare by loading in the embedding
    embed = load()

    # Now, split the data into chunks to be processed in parallel
    counter = 1
    for x in np.array_split(data, num_threads, axis=0):
        x.to_csv(f"{tmp_dir}datasplit_{counter}.csv")
        counter += 1

    del data
    gc.collect()

    # Launch threads to process each file
    threads = []
    for counter in range(1, num_threads + 1):
        t = mp.Process(target=average_similarity_worker, args=(counter, embed, ref_data, f"{tmp_dir}datasplit_{counter}.csv", f"{tmp_dir}resultsplit_{counter}.csv"))
        threads.append(t)
        t.start()

    # Join the threads back
    for thread in threads:
        thread.join()

    '''# This functionality generates a file that is too large to process, so it has been removed
    # Combine the result files & write them to the save location
    for counter in range(1, num_threads + 1):
        df = pd.read_csv(f"{tmp_dir}resultsplit_{counter}.csv", dtype=str, keep_default_na=False)
        if counter == 1:
            pd.DataFrame(columns=df.columns).to_csv(save_loc, index=False)
        df.to_csv(save_loc, mode="a", header=False, index=False)

    # Re-read the data and reformat it for later use
    data = pd.read_csv(save_loc, dtype=str, keep_default_na=False)
    # Break the embedded description's array column into multiple columns
    new_data = data["desc"].transform(lambda x: np.asarray(x[1:-1].split(', '), "float32")).apply(pd.Series)
    # Rename the new columns & add in the ID & combined column again
    embed_data = new_data.rename(lambda x: f"embed_{x}", axis=1)
    # Break the cosine similarity column into multiple columns
    new_data = data["cos_sim"].transform(lambda x: np.asarray(x[1:-1].split(', '), "float32")).apply(pd.Series)
    embed_data = pd.concat([embed_data, new_data.rename(lambda x: f"cos_sim_{x}", axis=1)], axis=1)
    # Break the euclidean similarity into multiple columns
    new_data = data["euc_sim"].transform(lambda x: np.asarray(x[1:-1].split(', '), "float32")).apply(pd.Series)
    embed_data = pd.concat([embed_data, new_data.rename(lambda x: f"euc_sim_{x}", axis=1)], axis=1)
    # Add in the ID and combined columns
    embed_data["id"] = data["id"]
    #embed_data["combined_embed"] = data["desc"]
    #embed_data["combined_cos_sim"] = data["cos_sim"]
    #embed_data["combined_euc_sim"] = data["euc_sim"]
    # Write the data to the file again
    embed_data.to_csv(save_loc, index=False)
    '''

    # Read in each result file, reformat it, and save them
    filename_split = save_loc.split(".")
    filename_split = [".".join(filename_split[:-1]), filename_split[-1]]
    threads = []
    print("GLOVE-SIM: Formatting embeddings and similarities...")
    for counter in range(1, num_threads+1):
        #t = mp.Process(target=similarity_reformat_worker, args=(counter, f"{tmp_dir}resultsplit_{counter}.csv", f"{tmp_dir}convertsplit_{counter}.csv"))
        t = mp.Process(target=similarity_reformat_worker, args=(counter, f"{tmp_dir}resultsplit_{counter}.csv", f"{filename_split[0]}_{counter}.{filename_split[1]}"))
        threads.append(t)
        t.start()
    for thread in threads:
        thread.join()

    # Combine the result files & write them to the save location
    for counter in tqdm(range(1, num_threads + 1), desc="GLOVE-SIM: Combining result files..."):
        #df = pd.read_csv(f"{tmp_dir}convertsplit_{counter}.csv", dtype=str, keep_default_na=False)
        df = pd.read_csv(f"{filename_split[0]}_{counter}.{filename_split[1]}", dtype=str, keep_default_na=False)
        if counter == 1:
            pd.DataFrame(columns=df.columns).to_csv(save_loc, index=False)
        df.to_csv(save_loc, mode="a", header=False, index=False)

    # Remove all temporary files
    for counter in range(1, num_threads + 1):
        os.remove(f"{tmp_dir}datasplit_{counter}.csv")
        os.remove(f"{tmp_dir}resultsplit_{counter}.csv")
        #os.remove(f"{tmp_dir}convertsplit_{counter}.csv")


def average_similarity_worker(number, embedding, ref_data, data_loc, save_loc):
    CHUNK_SIZE = 2500 # Add this to config later
    # For each chunk, do this processing
    prev_len = CHUNK_SIZE
    chunk_counter = 0
    while chunk_counter < 2:#prev_len == CHUNK_SIZE:
        # Read the csv
        if not chunk_counter:
            df = pd.read_csv(data_loc, dtype=str, keep_default_na=False, nrows=CHUNK_SIZE)
        else:
            df = pd.read_csv(data_loc, dtype=str, header=None, names=["index", "id", "desc"], keep_default_na=False, nrows=CHUNK_SIZE, skiprows=(CHUNK_SIZE*(chunk_counter)))
        df.drop(columns=df.columns[0], inplace=True)
        #df.reset_index(inplace=True)
        prev_len = len(df)

        # If this is the first loop, write the initial header
        columns = np.append(np.asarray(df.columns[0:2]), ["cos_sim", "euc_sim"])
        if not chunk_counter:
            pd.DataFrame(columns=columns).to_csv(save_loc, index=False)

        # For each entry, get the average embedding
        temp_embed = []
        for i, row in tqdm(df.iterrows(), desc=f"THREAD-{number}: Calculating averaged GloVe embeddings... (Chunk {chunk_counter+1})"):
            # Calculate the embedding
            embed = get_avg_text_vector(row["desc"], embedding)
            # Append the averaged text vector
            temp_embed.append({
                "id": row["id"],
                "desc": embed,
                "cos_sim": ref_data["embed"].transform(lambda x: distance.cosine(embed, x)).tolist(),
                "euc_sim": ref_data["embed"].transform(lambda x: distance.euclidean(embed, x)).tolist()
            })
        chunk_counter += 1

        # Write the result to the file
        pd.DataFrame(temp_embed, columns=columns).to_csv(save_loc, mode="a", header=False, index=False)


def nonbatched_average_similarity_worker(number, embedding, ref_data, data_loc, save_loc):
    # Read the data file
    df = pd.read_csv(data_loc, dtype=str, keep_default_na=False)
    df.drop(columns=df.columns[0], inplace=True)

    # Write the initial header to the file
    columns = np.append(np.asarray(df.columns[0:2]), ["cos_sim", "euc_sim"])
    pd.DataFrame(columns=columns).to_csv(save_loc, index=False)

    # For each entry in the file, get the average embedding, cosine similarity, and euclidean similarity
    temp_embed = []
    for i, row in tqdm(df.iterrows(), desc=f"THREAD-{number}: Calculating averaged GloVe embeddings..."):
        # Calculate the embedding
        embed = get_avg_text_vector(row["desc"], embedding)
        # Append the averaged text vector
        temp_embed.append({
            "id": row["id"],
            "desc": embed,
            "cos_sim":ref_data["embed"].transform(lambda x: distance.cosine(embed, x)).tolist(),
            "euc_sim":ref_data["embed"].transform(lambda x: distance.euclidean(embed, x)).tolist()
        })

    # Write the result to the file
    pd.DataFrame(temp_embed, columns=columns).to_csv(save_loc, mode="a", header=False, index=False)


def similarity_reformat_worker(number, data_loc, save_loc):# This function breaks the generation, so split the processing into chunks. Super easy
    CHUNK_SIZE = 2500  # Add this to config later
    # For each chunk, do this processing
    prev_len = CHUNK_SIZE
    chunk_counter = 0
    columns = []
    while prev_len == CHUNK_SIZE:
        # Read the csv
        if not chunk_counter:
            df = pd.read_csv(data_loc, dtype=str, keep_default_na=False, nrows=CHUNK_SIZE)
            columns = df.columns
        else:
            df = pd.read_csv(data_loc, dtype=str, header=None, names=columns, keep_default_na=False,
                             nrows=CHUNK_SIZE, skiprows=(CHUNK_SIZE * (chunk_counter)))
        #df.drop(columns=df.columns[0], inplace=True)
        # df.reset_index(inplace=True)
        prev_len = len(df)

        # If this is the first loop, write the initial header
        columns = np.append(np.asarray(df.columns[0:2]), ["cos_sim", "euc_sim"])
        if not chunk_counter:
            pd.DataFrame(columns=columns).to_csv(save_loc, index=False)

        # Reformat the dataframe and write it
        # Break the embedded description into multiple columns
        new_data = df["desc"].transform(lambda x: np.asarray(x[1:-1].split(', '), "float32")).apply(pd.Series)
        # Rename the new columns and create the resultant dataframe
        result_data = pd.concat([df["id"], new_data.rename(lambda x: f"embed_{x}", axis=1)], axis=1)
        # Break the cosine similarity column into multiple columns
        new_data = df["cos_sim"].transform(lambda x: np.asarray(x[1:-1].split(', '), "float32")).apply(pd.Series)
        result_data = pd.concat([result_data, new_data.rename(lambda x: f"cos_sim_{x}", axis=1)], axis=1)
        # Break the euclidean similarity into multiple columns
        new_data = df["euc_sim"].transform(lambda x: np.asarray(x[1:-1].split(', '), "float32")).apply(pd.Series)
        result_data = pd.concat([result_data, new_data.rename(lambda x: f"euc_sim_{x}", axis=1)], axis=1)

        # Write the result to the file
        print(f"GLOVE-SIM: File {number} chunk {chunk_counter+1} has finished processing. Writing it to disk")
        if not chunk_counter:
            result_data.to_csv(save_loc, index=False)
        else:
            result_data.to_csv(save_loc, mode="a", header=False, index=False)
        print(f"GLOVE-SIM: File {number} chunk {chunk_counter+1} has been written")
        chunk_counter += 1
    print(f"GLOVE-SIM: File {number} has finished processing and has been written to disk")
    #-------------------------------------------------------------------------------------------------------------------
    '''# Read in the data
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

    print(f"GLOVE-SIM: File {number} has finished processing. Writing it to disk")
    # Write the data to a file
    result_data.to_csv(save_loc, index=False)
    print(f"GLOVE-SIM: File {number} has been written")'''


def nonbatched_similarity_reformat_worker(number, data_loc, save_loc):
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

    print(f"GLOVE-SIM: File {number} has finished processing. Writing it to disk")
    # Write the data to a file
    result_data.to_csv(save_loc, index=False)
    print(f"GLOVE-SIM: File {number} has been written")