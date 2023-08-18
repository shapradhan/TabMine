import json
import os
import pandas as pd

from text_preprocessor import process_texts
from utils.embeddings import get_embeddings_dict, calculate_similarity_between_embeddings


def calculate_similarities(community_labels, embedding_model, dkd_file_path, dkd_embeddings_dir, labels_embeddings_dir):
    """ Calculate the similarity between community labels and documents from the Domain Knowledge Definion (DKD) file.

    Args:
        community_labels (dict): A dictionary where keys are the community IDs and values are the corresponding community labels.
        embedding_model (callable): A callable object representing the embedding model. The model must take
            a list of input strings and return a list of embeddings for those strings.
        dkd_file_path (str): The path of the DKD file.
        dkd_embeddings_dir (str): The name of the directory in which embeddings of the DKD documents are stored.
        labels_embeddings_dir (str): The name of the directory in which emebddings of the community labels are stored.
    
    Returns:
        pd.DataFrame: A DataFrame containing columns for the document, the label, and the similarity score between them.
    """

    if os.path.exists(dkd_file_path):
        with open(dkd_file_path, 'r') as json_file:
            data = json.load(json_file)

            documents_dict = {val: process_texts(val) for val in data['documents']}
            labels_dict = {val: val for val in community_labels.values()}
     
            document_embeddings_dict = get_embeddings_dict(documents_dict, embedding_model, dkd_embeddings_dir)
            labels_embeddings_dict = get_embeddings_dict(labels_dict, embedding_model, labels_embeddings_dir)

            results = []
            for k1, v1 in document_embeddings_dict.items():
                for k2, v2 in labels_embeddings_dict.items():
                    sim_score = calculate_similarity_between_embeddings(v1, v2)
                    results.append({
                        'document': k1,
                        'label': k2,
                        'similarity_score': sim_score
                    })
            
            df = pd.DataFrame(results)
            return df

    else:
        raise FileNotFoundError("Given file not found.")