import pandas as pd
import spacy
import tensorflow_hub as hub

from embeddings_utils import get_string_embedding, save_embeddings_to_file, load_embeddings_from_file, \
    calculate_average_similarity_of_embeddings, compute_average_embedding, calculate_similarity_between_embeddings
from utils import is_file_in_subfolder

def move_when_even_item_numbers(left_nodes, right_nodes, embeddings_dict):
    """Moves a node according to its similarity to neighboring nodes
    
    Args:
        left_nodes (list): A list of nodes
        right_nodes (list): A list of nodes
        embeddings_dict (dict): A dictionary that contains the embeddings

    Returns:
        list, list: left nodes and right nodes after their nodes have been moved
    """

    if len(left_nodes) >= 2 and len(right_nodes) >= 2:
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
        #   1. Last item of left nodes and first item of right noes
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


    embeddings_list = embeddings_array.tolist()
    return embeddings_list

    

    new_list = []
    for node_list in full_node_list:
        node_list_embeddings = create_sentence_embeddings(node_list, embeddings_identifier='node_list')
        node_list_embeddings_average_similarity = calculate_average_similarity(node_list_embeddings)
        sim_threshold = 0.8

        if node_list_embeddings_average_similarity < sim_threshold:
            all_above_threshold = False

            num_nodes = len(node_list)
            cut_off_index = num_nodes // 2
            if num_nodes % 2 == 0:
                left_nodes = node_list[:cut_off_index]
                right_nodes = node_list[cut_off_index:]
            else:
                left_nodes = node_list[:cut_off_index]
                right_nodes = node_list[cut_off_index+1:]
                mid_node = node_list[cut_off_index]
                mid_node_description = table_description_dict[mid_node]
                mid_node_embeddings = create_sentence_embeddings(text_list=[mid_node_description], embeddings_identifier='mid_node')
            left_nodes_descriptions = [table_description_dict[node] for node in left_nodes]
            right_nodes_descriptions = [table_description_dict[node] for node in right_nodes]
            left_nodes_embeddings = create_sentence_embeddings(text_list=left_nodes_descriptions, embeddings_identifier='contrived_2_left_nodes')
            right_nodes_embeddings = create_sentence_embeddings(text_list=right_nodes_descriptions, embeddings_identifier='contrived_2_right_nodes')

            left_nodes_avg_embeddings = calculate_average_embedding(left_nodes_embeddings)
            right_nodes_avg_embeddings = calculate_average_embedding(right_nodes_embeddings)

            if num_nodes % 2 == 0:
                pass
            else:
                mid_left_similarity_score = calculate_cosine_similarity(mid_node_embeddings, left_nodes_avg_embeddings)
                mid_right_similarity_score = calculate_cosine_similarity(mid_node_embeddings, right_nodes_avg_embeddings)

                if mid_left_similarity_score >= mid_right_similarity_score:
                    left_nodes.append(mid_node)
                else:
                    right_nodes.append(mid_node)

                new_list.append(left_nodes)
                new_list.append(right_nodes)
        else:
            if len(new_list) == 0:
                new_list = node_list
            else:
                new_list.append(node_list)
    return new_list
if __name__ == '__main__':
    nlp = spacy.load('en_core_web_lg')

    df = pd.read_csv('contrived_descriptions.csv')
    # Contrived
    full_nodes_list = ['sales_1', 'sales_2', 'sales_flow_1', 'sales_flow_2', 'bill_1', 'bill_2', 'delivery_1']

    full_nodes_desc = [df.loc[df['tables'] == node, 'descriptions'].iloc[0] for node in full_nodes_list]

    dict_data_records = df.to_dict(orient='records')
    table_description_dict = {item['tables']: item['descriptions'] for item in dict_data_records}

    
    # Calculate the average embeddings similarity in the complete sequence
    full_nodes_embeddings = load_or_create_embeddings(filename='contrived_2_full_nodes_embeddings.npy', embeddings_identifier='contrived_2_full_nodes', nodes=full_nodes_desc)
    full_nodes_embeddings_average_similarity = calculate_average_similarity(full_nodes_embeddings)
    sim_threshold = 0.8

    if full_nodes_embeddings_average_similarity < sim_threshold:
        all_above_threshold = False

    while not all_above_threshold:
        full_nodes_list = move_until_above_threshold([full_nodes_list], table_description_dict)
        full_nodes_list = move_until_above_threshold(full_nodes_list, table_description_dict)
