from utils.general import is_file_in_subfolder
from utils.graph import get_nodes_in_community
from utils.data_dictionary import get_table_descriptions
from utils.embeddings import load_embeddings_from_file, create_string_embeddings, save_embeddings_to_file, \
    compute_average_embedding, calculate_similarity_between_embeddings

def get_embeddings_list(description_dict, embedding_model, embeddings_folder_name):
    """
    Generate embeddings for a list of descriptions using a specified model.

    Args:
        description_dict (dict): A dictionary where keys are tables and values are descriptions.
        embedding_model (tensorflow.python.saved_model.load.Loader._recreate_base_user_object.<locals>._UserObject): The pre-trained 
            embedding model used to generate embeddings.
        embeddings_folder_name (str)

    Returns:
        list: A list of embeddings, each corresponding to a description in the input dictionary.
    """

    embeddings_list = []
    for table, description in description_dict.items():
        embeddings_filename = embeddings_filename = '{0}_embeddings.npy'.format(table)

        if is_file_in_subfolder(embeddings_folder_name, embeddings_filename):
            embeddings = load_embeddings_from_file(embeddings_folder_name, embeddings_filename)
        else:
            embeddings = create_string_embeddings(description, embedding_model)
            save_embeddings_to_file(embeddings, folder=embeddings_folder_name, filename=embeddings_filename)

        embeddings_list.append(embeddings)

    return embeddings_list

def enriched_community_detector(df, partition, connecting_nodes_list, embedding_model):
    """Detect natural language-enriched communities and nodes using a specified model.

    Args:
        df (pandas.DataFrame): The DataFrame containing tables and descriptions.
        partition (dict): A dictionary where keys are nodes and values are the corresponding community IDs. 
                          The communities are numbered from 0 to number of communities.
        connecting_nodes_list (list): A list of tuples representing nodes that are connected with each other.
        embedding_model (tensorflow.python.saved_model.load.Loader._recreate_base_user_object.<locals>._UserObject): The pre-trained 
            embedding model used to generate embeddings.

    Returns:
        dict: A dictionary where keys are connecting nodes and values are the ID of the community in which it belongs to after
            they have been moved, if necessary.
    
    Raise:
        ValueError: This error si raised if there is no description in the connecting nodes list item.
    """

    EMBEDDINGS_FOLDER_NAME = 'embeddings'
    for key, val in connecting_nodes_list.items():
        if val:
            # Find the connecting node
            connecting_node = list(val)[0]
            connecting_nodes_description = df.loc[df['tables'] == connecting_node, 'descriptions'].iloc[0]

            # Get the neighboring nodes group sand current nodes group of the connecting node
            neighboring_nodes_original = get_nodes_in_community(partition, key[0])
            current_nodes_original = get_nodes_in_community(partition, key[1])
            
            # Current node group without the connecting node
            current_nodes_modified = current_nodes_original[:]    
            current_nodes_modified.remove(connecting_node)

            # Get descriptions of the nodes
            neighboring_nodes_original_descriptions = get_table_descriptions(df, neighboring_nodes_original)
            current_nodes_modified_descriptions = get_table_descriptions(df, current_nodes_original)

            # Create embeddings for each node group and the connecting node
            neighboring_nodes_original_embeddings = get_embeddings_list(neighboring_nodes_original_descriptions, embedding_model, EMBEDDINGS_FOLDER_NAME)
            current_nodes_modified_embeddings = get_embeddings_list(current_nodes_modified_descriptions, embedding_model ,EMBEDDINGS_FOLDER_NAME)
            connecting_node_embeddings = create_string_embeddings(connecting_nodes_description, embedding_model)
     
            # Compute the average embeddings for each node group
            neighboring_nodes_original_average_embeddings = compute_average_embedding(neighboring_nodes_original_embeddings)
            current_nodes_modified_average_embeddings = compute_average_embedding(current_nodes_modified_embeddings)

            # Calculat the similarity score between the connecting node and the node groups
            similarity_score_with_neighboring_nodes = calculate_similarity_between_embeddings(connecting_node_embeddings, neighboring_nodes_original_average_embeddings)
            similarity_score_with_current_nodes = calculate_similarity_between_embeddings(connecting_node_embeddings, current_nodes_modified_average_embeddings)

            # If the similarity score of the conencting node is higher with the neighboring node, move the node to that group
            # Else keep the node to the current group
            if similarity_score_with_neighboring_nodes > similarity_score_with_current_nodes:
                partition[connecting_node] = key[0]
            else:
                partition[connecting_node] = key[1]
        
    return partition