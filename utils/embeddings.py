import numpy as np
import os

from sklearn.metrics.pairwise import cosine_similarity

from utils.general import make_subdirectory, is_file_in_subdirectory


def create_string_embeddings(input_string, embedding_model):
    """Get embeddings for a given input string using the specified embedding model.

    Args:
        input_string (str): The input string for which embeddings are to be generated.
        embedding_model (callable): A callable object representing the embedding model. The model must take
            a list of input strings and return a list of embeddings for those strings.

    Returns: 
        tf.Tensor: A TensorFlow tensor containing the embeddings for the input string.
    """

    embeddings = embedding_model([input_string])
    return embeddings[0]


def save_embeddings_to_file(embeddings, folder_name, filename):
    """Save embeddings to a file in a given folder

    Args:
        embeddings (tf.Tensor):  A TensorFlow EagerTensor containing the embeddings to be saved.
        folder_name (str): The name of a folder where the file will be saved
        filename (str): The name of the file to be created.
    
    Returns:
        None
    """
    
    make_subdirectory(folder_name)
    file_path = os.path.join(folder_name, filename)
    np.save(file_path, embeddings)


def load_embeddings_from_file(subfolder_name, filename):
    """Load embeddings of a particular file stored in a subfolder
    
    Args:
        subfolder_name (str): The name of the subfolder
        filename (str): The name of the file in which the embeddings are stored

    Returns:
        numpy.ndarray: an array of embeddings
    """

    file_path = os.path.join(os.getcwd(), subfolder_name, filename)
    embeddings_array = np.load(file_path)
    return embeddings_array


def compute_average_embedding(embeddings):
    """Compute the avareage embedding from a list of embeddings
    
    Args:
        embeddings (list): A list of embeddings
    
    Returns:
        numpy.ndarray: A Numpy array representing the average embedding
    """

    if len(embeddings) == 0:
        raise ValueError("Input emebddings list is empty.")
    
    embeddings_array = np.array(embeddings)
    average_embedding = np.mean(embeddings_array, axis=0)   # Average embedding along the first axis (axis=0)
    return average_embedding

def calculate_average_similarity(embeddings):
    """ Calculate the average cosine similarity between pairs of embeddings.

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

    similarity_scores = cosine_similarity(embeddings)
    embeddings_length = len(embeddings)

    num_pairs = embeddings_length * (embeddings_length - 1) // 2  # Calculate the number of unique pairs
    
    total_similarity = (np.sum(similarity_scores) / 2) - (embeddings_length / 2)   # Exclude diagonal values (similarity to itself)
    average_similarity = total_similarity / num_pairs

    return average_similarity

def calculate_similarity_between_embeddings(embedding1, embedding2):
    """Calculate similarity score between two embeddings using cosine similarity
    
    Args:
        embeddings1 (list): A list of embeddings
        embeddings2 (list): A list of embeddings
    
    Returns:
        float or numpy.ndarray: If the input is 1D, returns the similarity value. If the input is 2D, returns an array of similarity values
    """
    
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)

    # Reshape the embeddings to be 2D arrays for cosine_similarity function
    if embedding1.ndim == 1:
        embedding1 = embedding1.reshape(1, -1)
    if embedding2.ndim == 1:
        embedding2 = embedding2.reshape(1, -1)

    similarity = cosine_similarity(embedding1, embedding2)

    # Since the input might be 1D, return the similarity value directly
    if similarity.size == 1:
        return similarity[0, 0]

    # If the input is 2D, return the 2D array of similarity values
    return similarity


def get_embeddings_dict(description_dict, model, embeddings_folder_name):
    """Create a dictionary of embeddings with table (node) name as the key and its embeddings as the value.

    Args:
        description_dict (dict): A dictionary with table (node) name as the key and its description as the value.
        embeddings_folder_name (str): The name of the folder in which the embeddings are stored.
        model (tensorflow.python.saved_model.load.Loader._recreate_base_user_object.<locals>._UserObject): The pre-trained 
            embedding model used to generate embeddings.
    
    Returns:
        dict: A dictionary with table (node) names as they key and its embeddings as the value.
    """

    embeddings_dict = {}

    # Traverse through the dictionary and create emebddings dictionary 
    for table, description in description_dict.items():
        embeddings_filename = '{0}_embeddings.npy'.format(table)

        # If embeddings file for a particular table exists, load the embeddings from file
        # If embeddings file does not exist for a particular table, create embeddings and save it in a file
        if is_file_in_subdirectory(embeddings_folder_name, embeddings_filename):
            embeddings = load_embeddings_from_file(embeddings_folder_name, embeddings_filename)
        else:
            embeddings = create_string_embeddings(description, model)
            save_embeddings_to_file(embeddings, folder_name=embeddings_folder_name, filename=embeddings_filename)

        embeddings_dict[table] = embeddings
    return embeddings_dict