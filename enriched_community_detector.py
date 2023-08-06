from utils.embeddings import calculate_similarity_between_embeddings, compute_average_embedding
from utils.general import find_key_by_value, get_multiple_occuring_values
from utils.graph import get_nodes_in_community


def compute_similarity_between_node_and_node_group(node, node_group, embeddings_dict):
    """ Compute similarity score between a given node and a node group.

    Args:
        node (str): The given node for which the similarity score has to be computed.
        node_group (list): A list of nodes.
        embeddings_dict (dict): A dictionary with table (node) name as the key and the embeddings of its descriptions as the value.
    
    Returns:
        float: The similarity between the given node and the node group
    """
    
    node_embeddings = embeddings_dict[node]
    node_group_embeddings = [embeddings_dict[node] for node in node_group]
    node_group_average_embeddings = compute_average_embedding(node_group_embeddings)
    return calculate_similarity_between_embeddings(node_embeddings, node_group_average_embeddings)


def enriched_community_detector(partition, nodes_by_community, community_connecting_nodes_dict, embeddings_dict):
    """Detect natural language-enriched communities and nodes using a specified model.

    Args:
        partition (dict): A dictionary where keys are nodes and values are the corresponding community IDs. 
            The communities are numbered from 0 to number of communities.
        nodes_by_community (dict): A dictionary where keys are the community IDs and values are the list of nodes in respective communities.
        community_connecting_nodes_dict (dict): A dictionary where keys are tuples connecting two values representing neighboring 
            communities. The values are the set of nodes connecting the two communities represented by the tuple. The set contains
            either 0 or 1 item.
        embedding_model (tensorflow.python.saved_model.load.Loader._recreate_base_user_object.<locals>._UserObject): The pre-trained 
            embedding model used to generate embeddings.

    Returns:
        dict: A dictionary where keys are connecting nodes and values are the ID of the community in which it belongs to after
            they have been moved, if necessary.
    
    Raises:
        ValueError: This error si raised if there is no description in the connecting nodes list item.

    Example:
        connecting_node_list = {
            (1, 0): {'vbfa'}, (1, 3): set(), (1, 2): set(), (0, 1): {'vbrp'}, 
            (0, 3): {'vbak'}, (0, 2): set(), (3, 1): set(), (3, 0): {'vbap'}, 
            (3, 2): set(), (2, 1): set(), (2, 0): set(), (2, 3): set()
        }
    """

    # Identify the nodes connecting two communities from community_connecting_nodes_dict
    # For above example, nodes_connecting_two_communities = ['vbfa', 'vbrp', 'vbak', 'vbap']
    nodes_connecting_two_communities = [item for subset in community_connecting_nodes_dict.values() for item in subset]

    # If there is connecting more than two communities, identify such nodes
    # E.g., if nodes_connecting_two_communities = ['vbfa', 'vbrp', 'vbak', 'vbfa'],
    # nodes_connecting_multiple_communities = ['vbfa']
    nodes_connecting_multiple_communities = get_multiple_occuring_values(nodes_connecting_two_communities)

    # If there is a node that connects more than two communities, then find all neighboring communities of that node
    if nodes_connecting_multiple_communities:
        neighboring_node_groups = []
        for key, val in community_connecting_nodes_dict.items():
            if val:
                connecting_node = list(val)[0]  # Example: val = {'vbfa'}. So, connecting_node becomes 'vbfa'

                # If connecting node is in nodes_connecting_multiple_communities, get the neighboring nodes.
                # Append the nodes of all neighboring communities to a a list.
                # Remove the reference of the node being a node connecting two communities from community_connecting_nodes_dict.
                # The removal is done so that the algorithm would not have to traverse through this node again.
                if connecting_node in nodes_connecting_multiple_communities:
                    neighboring_nodes_original = get_nodes_in_community(partition, key[0])
                    neighboring_node_groups.append(neighboring_nodes_original)
                    community_connecting_nodes_dict[key] = set()
        
        similarity_scores = []    
        
        # For each neighboring node group, compute the similarity between that group and the connecting node.
        # Append the similarity scores to a list.
        for neighboring_nodes in neighboring_node_groups:
            similarity_score = compute_similarity_between_node_and_node_group(connecting_node, neighboring_nodes, embeddings_dict)
            similarity_scores.append(similarity_score)

        # Determine which community to move the connecting node to.
        # This is done by comparing the similarity scores with each other.
        # Here, an assumption is that connecting node only connects at most three communities - its own and two neighboring communities.
        community_to_move_to = -1
        if similarity_scores:
            if similarity_scores[0] > similarity_scores[1]:
                community_to_move_to = find_key_by_value(nodes_by_community, neighboring_node_groups[0])
            else:
                community_to_move_to = find_key_by_value(nodes_by_community, neighboring_node_groups[1])
                
            partition[connecting_node] = community_to_move_to

    # Iterate through rest of the nodes connecting the communities
    for key, val in community_connecting_nodes_dict.items():
        if val:
            connecting_node = list(val)[0]  # Example: val = {'vbfa'}. So, connecting_node becomes 'vbfa'

            # Get the neighboring nodes group and current nodes group of the connecting node
            neighboring_nodes_original = get_nodes_in_community(partition, key[0])
            current_nodes_original = get_nodes_in_community(partition, key[1])
        
            # Current node group without the connecting node
            current_nodes_modified = current_nodes_original[:]    
            current_nodes_modified.remove(connecting_node)
            
            # Compute the similarity score between the connecting node and its neighboring node group and its own node group without itself
            similarity_score_with_neighboring_nodes = compute_similarity_between_node_and_node_group(connecting_node, neighboring_nodes_original, embeddings_dict)
            similarity_score_with_current_nodes = compute_similarity_between_node_and_node_group(connecting_node, current_nodes_modified, embeddings_dict)
            
            # If the similarity score of the conencting node is higher with the neighboring node, move the node to that group
            # Else keep the node to the current group
            if similarity_score_with_neighboring_nodes > similarity_score_with_current_nodes:
                partition[connecting_node] = key[0]
            else:
                partition[connecting_node] = key[1]
        
    return partition