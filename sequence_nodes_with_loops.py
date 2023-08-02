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

def move_until_above_threshold(full_node_list, embeddings_dict):
    """Move the nodes until a threshold is met

    Args:
        fill_node_list (list): A list of nodes
        embeddings_dict (dict): A dictionary that contains the embeddings
    
    Returns:
        list: A list containing nodes that have been moved
    """

    new_list = []
    EMBEDDINGS_FOLDER_NAME = 'embeddings'
    for node_list in full_node_list:
        node_list_embeddings = []
        
        # For a group of nodes (node_list), get embeddings
        for i in node_list:
            embeddings_filename = '{0}_embeddings.npy'.format(i)
            embeddings = load_embeddings_from_file(EMBEDDINGS_FOLDER_NAME, embeddings_filename)
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
            

if __name__ == '__main__':
    nlp = spacy.load('en_core_web_lg')
    EMBEDDINGS_FOLDER_NAME = 'embeddings'

    df = pd.read_csv('contrived_descriptions.csv')

    # seq_nodes = ['vbrp', 'vbrk', 'bkpf', 'bseg']
    # seq_nodes = ['vbak', 'vbap', 'vbfa', 'lips', 'likp']

    # Contrived example
    full_nodes_list = ['sales_1', 'sales_2', 'sales_flow_1', 'sales_flow_2', 'bill_1', 'delivery_1', 'delivery_2', 'delivery_3', 'purchase_1', 'purchase_2']
    full_nodes_desc = [df.loc[df['tables'] == node, 'descriptions'].iloc[0] for node in full_nodes_list]
    
    # Create dictionary of descriptions
    dict_data_records = df.to_dict(orient='records')
    table_description_dict = {item['tables']: item['descriptions'] for item in dict_data_records}
    
    # Load the embeddings model
    model_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    model = hub.load(model_url)

    embeddings_dict = {}
    full_nodes_embeddings = []

    # Traverse through the dictionary and create dictionary and list of embeddings
    for table, description in table_description_dict.items():
        embeddings_filename = '{0}_embeddings.npy'.format(table)

        # If embeddings file for a particular table exists, load the embeddings from file
        # If embeddings file does not exist for a particular table, create embeddings and save it in a file
        if is_file_in_subfolder(EMBEDDINGS_FOLDER_NAME, embeddings_filename):
            embeddings = load_embeddings_from_file(EMBEDDINGS_FOLDER_NAME, embeddings_filename)
        else:
            embeddings = get_string_embedding(description, model)
            save_embeddings_to_file(embeddings, folder=EMBEDDINGS_FOLDER_NAME, filename=embeddings_filename)

        embeddings_dict[table] = embeddings
        full_nodes_embeddings.append(embeddings)
    
    all_above_threshold = False
    while not all_above_threshold:
        full_nodes_embeddings_average_similarity = calculate_average_similarity_of_embeddings(full_nodes_embeddings)
        sim_threshold = 0.8
    
        full_nodes_list = move_until_above_threshold([full_nodes_list], embeddings_dict)
        print('FNL 1:', full_nodes_list)
        full_nodes_list = move_until_above_threshold(full_nodes_list, embeddings_dict)
        print('FNL 2:', full_nodes_list)
        full_nodes_list = move_until_above_threshold(full_nodes_list, embeddings_dict)
        print('FNL 3:', full_nodes_list)
        all_above_threshold = True