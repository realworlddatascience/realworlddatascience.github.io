# embedding.py
# This interface allows us to easily call embedding functions generically
# Auburn Big Data

# Internal Imports
from src.utilities.embedding import glove, bert
# External Imports


def get_avg_embeds(embed_name, data, save_loc, num_threads=0):
    if embed_name == "bert":
        return bert.get_avg_embeds(data, save_loc, num_threads)
    else:
        return glove.get_average_embeddings(data, save_loc, num_threads)


def get_avg_sim(embed_name, data, reference, save_loc, num_threads=0):
    if embed_name == "bert":
        return bert.get_avg_sim(data, reference, save_loc, num_threads)
    else:
        return glove.get_average_similarities(data, reference, save_loc, num_threads)