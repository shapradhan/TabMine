import numpy as np
from os import getenv
from sklearn.metrics.pairwise import cosine_similarity

from text_embedder import TextEmbedder
from text_preprocessor import TextPreprocessor
from utils.general import is_file_in_subdirectory, read_lines

def get_embeddings_dict(table_name, description, model, embeddings_dict):
    """
    Create a dictionary of embeddings with table (node) name as the key and its embeddings as the value.

    Args:
        description_dict (dict): A dictionary with table (node) name as the key and its description as the value.
        embeddings_folder_name (str): The name of the folder in which the embeddings are stored.
        model (tensorflow.python.saved_model.load.Loader._recreate_base_user_object.<locals>._UserObject): The pre-trained 
            embedding model used to generate embeddings.
    
    Returns:
        dict: A dictionary with table (node) names as they key and its embeddings as the value.
    """
    description = description.lower()

    preprocessed_texts = {}

    PROCESS_RAW = getenv('PROCESS_RAW').lower() in ['true', 'yes', 1]
    POS_TAGGED = getenv('POS_TAGGED').lower() in ['true', 'yes', 1]
    NOUNS_ONLY = getenv('NOUNS_ONLY').lower() in ['true', 'yes', 1]
    TABLE_NAME_INCLUDED = getenv('TABLE_NAME_INCLUDED').lower() in ['true', 'yes', 1]
    EMBEDDINGS_DIR = getenv('EMBEDDINGS_DIR')

    embeddings_filename = '{0}_embeddings.npy'.format(table_name)

    if is_file_in_subdirectory(EMBEDDINGS_DIR, embeddings_filename):
        embedder = TextEmbedder()
        embeddings = embedder.load_embeddings_from_file(EMBEDDINGS_DIR, embeddings_filename)
    else:
        if TABLE_NAME_INCLUDED:
            description = table_name + " " + description
            
        if PROCESS_RAW:
            embedder = TextEmbedder(description, model)
        else:
            common_terms = read_lines('common_terms.txt')
            preprocessed_text = TextPreprocessor(description).preprocess(common_terms, POS_TAGGED, NOUNS_ONLY)
            preprocessed_texts[table_name] = preprocessed_text
            embedder = TextEmbedder(preprocessed_text, model)
            
        embeddings = embedder.create_embeddings()
        embedder.save_embeddings(embeddings, EMBEDDINGS_DIR, embeddings_filename)
    
    embeddings_dict[table_name] = embeddings
    return embeddings_dict

def calculate_average_similarity(embeddings):
    """ 
    Calculate the average cosine similarity between pairs of embeddings.

    Args:
        embeddings (list): List of embedding vectors.

    Returns:
        float: Average cosine similarity.

    Example:
        If emebddings list contains embeddings for 4 items A, B, C, and D.
        similarity_scores matrix will look like the following.
              A        B      C       D
          A  (1.00)  (0.34)  (0.23)  (0.54)
          B  0.34    (1.00)  (0.27)  (0.59)
          C  0.23     0.27   (1.00)  (0.24)
          D  0.54     0.59    0.24   (1.00)
        We don't want the values in brackets because the similarity between A and B, for instance, is the same as similarity between B and A.
        We also don't want the similarity between self nodes (e.g., A and A).

        num_pairs = embeddings_length * embeddings_length - 1) // 2 = 4 * (4 - 1) // 2 = 6 (AB, AC, AD, BC, BD, CD)

        For total_similarity = (np.sum(similarity_scores) / 2) - (embeddings_length / 2):
            np.sum(similarity_scores) = 8.42 (which includes 1+1+1+1 of self-similarity between AA, BB, CC, and DD)
            np.sum(similarity_scores) / 2 = 4.21 (this includes half of the self-similarity values)
            Subtracting (embeddings_length / 2) or (4/2 = 2) removes the half of the self-similarity values
        Therefore, average_similarity = 2.105 / 6 = 0.351
    """

    embeddings = np.vstack(embeddings)

    similarity_scores = cosine_similarity(embeddings)

    embeddings_length = len(embeddings)
    num_pairs = embeddings_length * (embeddings_length - 1) // 2  # Calculate the number of unique pairs
   
    total_similarity = (np.sum(similarity_scores) / 2) - (embeddings_length / 2)   # Exclude diagonal values (similarity to itself)
    average_similarity = total_similarity / num_pairs

    return average_similarity