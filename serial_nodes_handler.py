import pandas as pd

from utils.embeddings import create_string_embeddings, save_embeddings_to_file, load_embeddings_from_file, \
    calculate_average_similarity_of_embeddings, compute_average_embedding, calculate_similarity_between_embeddings
from utils.general import is_file_in_subfolder, all_values_higher_than

def move_when_even_item_numbers(left_nodes, right_nodes, embeddings_dict):
    """Move a node according to its similarity to neighboring nodes.
    
    Args:
        left_nodes (list): A list of nodes.
        right_nodes (list): A list of nodes.
        embeddings_dict (dict): A dictionary that contains the embeddings.

    Returns:
        list, list: left nodes and right nodes after their nodes have been moved.
    """

    # If left nodes and right nodes contain only one item each, return those items
    if len(left_nodes) == 1 and len(left_nodes) == 1:
        return left_nodes, right_nodes

    if len(left_nodes) > 1 and len(right_nodes) > 1:
        # Extract the last two nodes from the left nodes and first two nodes from the right nodes
        left_last_item = left_nodes[-1]
        left_second_last_item = left_nodes[-2]
        right_first_item = right_nodes[0]
        right_second_item = right_nodes[1]

        # Get embeddings of those nodes
        left_last_item_embeddings = embeddings_dict[left_last_item]
        left_second_last_item_embeddings = embeddings_dict[left_second_last_item]
        right_first_item_embeddings = embeddings_dict[right_first_item]
        right_second_item_embeddings = embeddings_dict[right_second_item]

        # Calculate similarity scores between
        #   1. Last item of left nodes and first item of right nodes
        #   2. Last item and second last item of left nodes
        #   3. First item and second item of right nodes
        left_last_right_first_similarity_score = calculate_similarity_between_embeddings(left_last_item_embeddings, right_first_item_embeddings)
        left_last_left_second_last_similarity_score = calculate_similarity_between_embeddings(left_last_item_embeddings, left_second_last_item_embeddings)  
        right_first_right_second_last_similarity_score = calculate_similarity_between_embeddings(right_first_item_embeddings, right_second_item_embeddings)   

        # If the similarity between the last item of left nodes and the first item of right nodes is higher than
        # the similarity between the last item and second last item of left nodes, then move the last item of left nodes to right nodes
        if left_last_right_first_similarity_score > left_last_left_second_last_similarity_score:
            left_nodes.pop()
            right_nodes.insert(0, left_last_item)
            return left_nodes, right_nodes
        
        # If the similarity between the last item of left nodes and the first item of right nodes is higher than
        # the similarity between the first item and second item of right nodes, then move the first item of right nodes to left nodes
        if left_last_right_first_similarity_score > right_first_right_second_last_similarity_score:
            left_nodes.append(right_first_item)
            right_nodes.pop(0)
            return left_nodes, right_nodes

        return left_nodes, right_nodes
