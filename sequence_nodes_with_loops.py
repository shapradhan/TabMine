import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

import tensorflow_hub as hub

def save_embeddings_to_file(embeddings, filename):
    print("saving embedding {0}".format(filename))
    embeddings_array = np.array(embeddings)
    np.save(filename, embeddings_array)

def create_sentence_embeddings(text_list, embeddings_identifier):
    model_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    model = hub.load(model_url)
    
    print("creating embedding {0}".format(embeddings_identifier))
    embeddings = model(text_list)
    save_embeddings_to_file(embeddings, "embeddings_{0}.npy".format(embeddings_identifier))
    
    return embeddings

def load_embeddings_from_file(filename):
    print('loading {0}'.format(filename))
    embeddings_array = np.load(filename)

    embeddings_list = embeddings_array.tolist()
    return embeddings_list

def calculate_average_embedding(embeddings):
    if len(embeddings) == 0:
        raise ValueError("Input emebddings list is empty.")
    
    embeddings_array = np.array(embeddings)
    average_embedding = np.mean(embeddings_array, axis=0)   # Average embedding along the first axis (axis=0)
    return average_embedding

def calculate_average_similarity(embeddings):
    if len(embeddings) <= 1:
        raise ValueError("Input embeddings list should have at least 2 elements.")

    embeddings_array = np.array(embeddings)
    similarity_matrix = cosine_similarity(embeddings_array)

    # Exclude self-similarity and calculate the average similarity
    n = len(embeddings)
    average_similarity = (np.sum(similarity_matrix) - n) / (n * (n - 1))

    return average_similarity

def is_file_in_current_folder(filename):
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, filename)
    return os.path.isfile(file_path)
