import numpy as np

from os import getenv
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, pairwise_distances

from text_embedder import TextEmbedder
from text_preprocessor import TextPreprocessor
from utils.general import is_file_in_subdirectory

def _get_embeddings_folder_name(preprocessing_options):
    """
    Generate a folder name for storing embeddings based on the model and preprocessing options.

    Args:
        model (str): The name or identifier of the model used for generating embeddings. This string will be prefixed to the folder name.
        preprocessing_options (dict): A dictionary of preprocessing options that affect the folder name. The dictionary can contain the following keys:
            - 'raw_description' (bool): If `True`, includes 'raw' in the folder name to indicate no preprocessing.
            - 'stopwords' (bool): If `True`, includes 'stopwords' in the folder name to indicate that stopwords removal was applied.
            - 'punctuations' (bool): If `True`, includes 'punctuations' in the folder name to indicate that punctuation removal was applied.
            - 'lemmatizing' (bool): If `True`, includes 'lemmatizing' in the folder name to indicate that lemmatization was applied.

    Returns:
        str: A string representing the folder name for storing embeddings, constructed based on the provided model and preprocessing options.

    Example:
        >>> get_embeddings_folder_name('bert', {'raw_description': True})
        'embeddings_bert_raw'
        
        >>> get_embeddings_folder_name('word2vec', {'stopwords': True, 'punctuations': True})
        'embeddings_word2vec_stopwords_punctuations'
        
        >>> get_embeddings_folder_name('gpt', {'lemmatizing': True})
        'embeddings_gpt_lemmatizing'
        
        >>> get_embeddings_folder_name('fasttext', {})
        'embeddings_fasttext'
    """

    # Start with the base folder name
    parts = ['embeddings_']

    # Append suffixes based on preprocessing options
    if preprocessing_options.get('raw_description', False):
        parts.append('raw')
    else:
        if preprocessing_options.get('stopwords', False):
            parts.append('stopwords')
        if preprocessing_options.get('punctuations', False):
            parts.append('punctuations')
        if preprocessing_options.get('lemmatizing', False):
            parts.append('lemmatizing')

    # Join parts
    embeddings_folder_name = '_'.join(parts)

    return embeddings_folder_name


def create_embeddings(table_name, description, embeddings_dict, preprocessing_options):
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

    DESCRIPTION_EMBEDDINGS_DIR = getenv('DESCRIPTION_EMBEDDINGS_DIR')
    TABLE_NAME_INCLUDED = getenv('TABLE_NAME_INCLUDED').lower() in ['true', 'yes', 1]
    POS_TAGGED = getenv('POS_TAGGED').lower() in ['true', 'yes', 1]
    NOUNS_ONLY = getenv('NOUNS_ONLY').lower() in ['true', 'yes', 1]

    embeddings_filename = table_name + '_embeddings.npy'
    embeddings_folder_name = _get_embeddings_folder_name(preprocessing_options)
    embeddings_dir_full_path = DESCRIPTION_EMBEDDINGS_DIR + '/' + embeddings_folder_name

    if is_file_in_subdirectory(embeddings_dir_full_path, embeddings_filename):
        embedder = TextEmbedder()
        embeddings = embedder.load_embeddings_from_file(embeddings_dir_full_path, embeddings_filename)
    else:
        if TABLE_NAME_INCLUDED:
            description = table_name + " " + description
        
        if preprocessing_options['raw_description']:
            embedder = TextEmbedder(description)
        else:
            preprocessed_text = TextPreprocessor(description).preprocess(preprocessing_options, POS_TAGGED, NOUNS_ONLY)
            preprocessed_texts[table_name] = preprocessed_text
            embedder = TextEmbedder(preprocessed_text)
            
        embeddings = embedder.create_embeddings()
        embedder.save_embeddings(embeddings, embeddings_dir_full_path, embeddings_filename)
    
    embeddings_dict[table_name] = embeddings
    return embeddings_dict

def _compute_similarity_matrix(embeddings, similarity_measure="cosine_similarity"):
    """
    Computes a similarity matrix from the given embeddings using the specified similarity measure.

    Args:
        embeddings (numpy.ndarray): A 2D array where each row represents an embedding vector for an item. 
            The shape of the array should be (n_samples, n_features).
        
        similarity_measure (str, optional, default="cosine_similarity"): The type of similarity measure to use for computing the similarity matrix. 
            Options include:
                - 'cosine_similarity': Computes the cosine similarity between embeddings.
                - 'euclidean_distance': Computes the Euclidean distance between embeddings.
                - 'manhatten': Computes the Manhattan distance (also known as L1 norm or cityblock distance) between embeddings.
                - 'dot_product': Computes the dot product between embeddings.

    Returns:
        numpy.ndarray: A 2D array representing the similarity matrix. The shape of the matrix is (n_samples, n_samples).
            - For 'cosine_similarity' and 'dot_product', the matrix represents similarity scores.
            - For 'euclidean_distance' and 'manhatten', the matrix represents distance scores.
        
    Raises:
    ValueError
        If an unsupported similarity_measure is provided.
    """

    similarity_map = {
        'cosine_similarity': lambda: cosine_similarity(embeddings),
        'euclidean_distance': lambda: euclidean_distances(embeddings),
        'manhattan_distance': lambda: pairwise_distances(embeddings, metric='cityblock'),
        'dot_product': lambda: np.dot(embeddings, embeddings.T)
    }
    
    if similarity_measure not in similarity_map:
        raise ValueError(f"Unsupported similarity_measure: '{similarity_measure}'. "
                         "Supported measures are: 'cosine_similarity', 'euclidean_distance', 'manhattan_distance', 'dot_product'.")
    
    # Execute the corresponding function
    return similarity_map[similarity_measure]()

def calculate_average_similarity(embeddings, similarity_measure="cosine_similarity"):
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
    similarity_matrix = _compute_similarity_matrix(embeddings, similarity_measure)
    
    embeddings_length = len(embeddings)
    num_pairs = embeddings_length * (embeddings_length - 1) // 2  # Calculate the number of unique pairs

    if similarity_measure == 'cosine_similarity' or similarity_measure == 'dot_product':
        total_similarity = (np.sum(similarity_matrix) / 2) - (embeddings_length / 2)   # Exclude diagonal values (similarity to itself)
    elif similarity_measure == 'euclidean_distance' or similarity_measure == 'manhattan_distance':
        total_similarity = (np.sum(similarity_matrix) / 2)

    average_similarity = total_similarity / num_pairs
    return average_similarity