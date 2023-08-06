import pandas as pd

from utils.embeddings import create_string_embeddings, save_embeddings_to_file, load_embeddings_from_file, \
    calculate_average_similarity_of_embeddings, compute_average_embedding, calculate_similarity_between_embeddings
from utils.general import is_file_in_subfolder, all_values_higher_than


def compare_average_similarity_and_threshold(node1, node2, similarity_threshold, embeddings_dict):
    """ Compare the average similarity of two nodes with the given similarity threshold.

    Args:
        node1 (str): A node.
        node2 (str): A node.
        similarity_threshold (float): The similarity threshold value.
        embeddings_dict (dict): A dictionary with table (node) name as the key and the embeddings of its descriptions as the value.
    
    Returns:
        bool: Return True if average similarity is higher than or equal to the similarity threshold. Otherwise, return False.
    """

    node1_embeddings = embeddings_dict[node1]
    node2_embeddings = embeddings_dict[node2]

    similarity_score = calculate_similarity_between_embeddings(node1_embeddings, node2_embeddings)

    return True if similarity_score >= similarity_threshold else False
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

def move_until_above_threshold(full_node_list, embeddings_dict, embeddings_folder_name):
    """Move the nodes until a threshold is met.

    Args:
        fill_node_list (list): A list of nodes.
        embeddings_dict (dict): A dictionary that contains the embeddings.
    
    Returns:
        list: A list containing nodes that have been moved.
    """

    new_list = []
    for node_list in full_node_list:
        node_list_embeddings = []
        
        # For a group of nodes (node_list), get embeddings
        for i in node_list:
            embeddings_filename = '{0}_embeddings.npy'.format(i)
            embeddings = load_embeddings_from_file(embeddings_folder_name, embeddings_filename)
            node_list_embeddings.append(embeddings)

        # Calculate the average similarity of the group of nodes
        node_list_embeddings_average_similarity = calculate_average_similarity_of_embeddings(node_list_embeddings)
        sim_threshold = 0.8

        # If the average similarity between a group of nodes is less than a given similarity threshold, divide the list into two subgroups
        if node_list_embeddings_average_similarity < sim_threshold:
            num_nodes = len(node_list)
            cut_off_index = num_nodes // 2

            # Handle even and odd number of items in node list
            if num_nodes % 2 == 0:
                left_nodes = node_list[:cut_off_index]
                right_nodes = node_list[cut_off_index:]
            else:
                left_nodes = node_list[:cut_off_index]
                right_nodes = node_list[cut_off_index+1:]
                mid_node = node_list[cut_off_index]
                mid_node_embeddings = embeddings_dict[mid_node]
            
            # Get embeddings for nodes in left and right groups   
            left_nodes_embeddings = [embeddings_dict[node] for node in left_nodes]
            right_nodes_embeddings = [embeddings_dict[node] for node in right_nodes]

            # Calculate the average of the embeddings of the nodes in the left and the right group
            left_nodes_avg_embeddings = compute_average_embedding(left_nodes_embeddings)
            right_nodes_avg_embeddings = compute_average_embedding(right_nodes_embeddings)

            # Move then nodes to the group where it is more similar
            if num_nodes % 2 == 0:
                left_nodes, right_nodes = move_when_even_item_numbers(left_nodes, right_nodes, embeddings_dict)
                new_list.append(left_nodes)
                new_list.append(right_nodes)
            else:
                mid_left_similarity_score = calculate_similarity_between_embeddings(mid_node_embeddings, left_nodes_avg_embeddings)
                mid_right_similarity_score = calculate_similarity_between_embeddings(mid_node_embeddings, right_nodes_avg_embeddings)
                
                # If similarity score of the mid node and left nodes group is higher than that of the mid node and right nodes group,
                # move the mid node to the left nodes. Otherwise, move it to the right nodes
                if mid_left_similarity_score >= mid_right_similarity_score:
                    left_nodes.append(mid_node)
                else:
                    right_nodes.insert(0, mid_node)
            
                new_list.append(left_nodes)
                new_list.append(right_nodes)
        else:
            new_list.append(node_list)        
    return new_list
