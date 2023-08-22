from utils.embeddings import calculate_similarity_between_embeddings, compute_average_embedding
from utils.general import get_multiple_occuring_values
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

def find_neighboring_nodes_in_same_community(partition, node_of_interest):
    """ Find neighboring nodes in the same community

    Args:
        partition (dict): A dictionary where keys are nodes and values are the corresponding community IDs. 
            The communities are numbered from 0 to number of communities.
        node_of_interest (str): The node for which we have to find neighboring nodes in the same community.
    
    Returns:
        list: A list containing the neighboring nodes of the node_of_interest in the same community.
    """
    
    node_community = partition[node_of_interest]
    neighbor_nodes_in_same_community = [node for node, community_id in partition.items() if community_id == node_community and node != node_of_interest]
    return neighbor_nodes_in_same_community
    """ Move connecting nodes to a more similar community 

    Args:
        partition (dict): A dictionary where keys are nodes and values are the corresponding community IDs. 
            The communities are numbered from 0 to number of communities.
        nodes_by_community (dict): A dictionary where keys are the community IDs and values are the list of nodes in respective communities.
        community_connecting_nodes_dict (dict): A dictionary where keys are tuples connecting two values representing neighboring 
            communities. The values are the set of nodes connecting the two communities represented by the tuple. The set contains
            either 0 or 1 item.
        embeddings_dict (dict): A dictionary with table (node) name as the key and the embeddings of its descriptions as the value.

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
    #nodes_by_community = {0: ['ekkn', 'ekes', 'eket', 'ekbe', 'ekko', 'ekpo'], 4: ['eina', 'eine'], 3: ['mkpf', 'mseg'], 5: ['eord'], 1: ['bkpf', 'bseg', 'rbkp', 'rseg'], 2: ['eban', 'ebkn']}
    # Identify the nodes connecting two communities from community_connecting_nodes_dict
    # For above example, nodes_connecting_two_communities = ['vbfa', 'vbrp', 'vbak', 'vbap']
    nodes_connecting_two_communities = [item for subset in community_connecting_nodes_dict.values() for item in subset]

    # If there is connecting more than two communities, identify such nodes
    # E.g., if nodes_connecting_two_communities = ['vbfa', 'vbrp', 'vbak', 'vbfa'],
    # nodes_connecting_multiple_communities = ['vbfa']
    nodes_connecting_multiple_communities = get_multiple_occuring_values(nodes_connecting_two_communities)
   
    # If there is a node that connects more than two communities, then find all neighboring communities of that node
    nodes_already_checked = []
    if nodes_connecting_multiple_communities:
        for node in set(nodes_connecting_multiple_communities):
            neighboring_community_nodes = {}
            current_communityId = None
            for edge, nodes in community_connecting_nodes_dict.items():
                if nodes:
                    nodes = list(nodes)
                    if node in nodes:
                        nodes_already_checked.append(node)
                        neighboring_community_nodes[edge[0]] = get_nodes_in_community(partition, edge[0])
                        current_communityId = edge[1]
            
            similarity_scores = {}
            for community_id, node_group in neighboring_community_nodes.items():
                sim_score = compute_similarity_between_node_and_node_group(node, node_group, embeddings_dict)
                similarity_scores[community_id] = sim_score

            current_community_nodes_original = get_nodes_in_community(partition, current_communityId)
            current_community_nodes_modified = current_community_nodes_original[:]
            current_community_nodes_modified.remove(node)
            
            intra_community_similarity = compute_similarity_between_node_and_node_group(node, current_community_nodes_modified, embeddings_dict)
            
            highest_similarity_score = intra_community_similarity
            community_with_highest_similarity_score = current_communityId
            for communityId, score in similarity_scores.items():
                if score > highest_similarity_score:
                    highest_similarity_score = score
                    community_with_highest_similarity_score = communityId   
                    
            partition[node] = community_with_highest_similarity_score

        neighboring_community_nodes = {}
        current_communityId = None
        for edge, nodes in community_connecting_nodes_dict.items():
            if nodes:
                nodes = list(nodes)
                for connecting_node in nodes:
                    if connecting_node not in nodes_already_checked:
                        neighboring_community_nodes = get_nodes_in_community(partition, edge[0])
                        current_community_nodes_original = get_nodes_in_community(partition, edge[1])
                        current_community_nodes_modified = current_community_nodes_original[:]
                        current_community_nodes_modified.remove(connecting_node)

                        intra_community_similarity = compute_similarity_between_node_and_node_group(connecting_node, current_community_nodes_modified, embeddings_dict)
                        similarity_score_with_neighboring_community = compute_similarity_between_node_and_node_group(connecting_node, neighboring_community_nodes, embeddings_dict)
                        
                        if intra_community_similarity < similarity_score_with_neighboring_community:
                            partition[connecting_node] = edge[0]
    else:
        for edge, nodes in community_connecting_nodes_dict.items():
            if nodes:
                nodes = list(nodes)
                for connecting_node in nodes:
                    # Get the neighboring nodes group and current nodes group of the connecting node
                    neighboring_nodes_original = get_nodes_in_community(partition, edge[0])
                    current_nodes_original = get_nodes_in_community(partition, edge[1])
                
                    # Current node group without the connecting node
                    current_nodes_modified = current_nodes_original[:]    
                    current_nodes_modified.remove(connecting_node)
                    
                    # Compute the similarity score between the connecting node and its neighboring node group and its own node group without itself
                    similarity_score_with_neighboring_nodes = compute_similarity_between_node_and_node_group(connecting_node, neighboring_nodes_original, embeddings_dict)
                    similarity_score_with_current_nodes = compute_similarity_between_node_and_node_group(connecting_node, current_nodes_modified, embeddings_dict)
                    
                    # If the similarity score of the conencting node is higher with the neighboring node, move the node to that group
                    # Else keep the node to the current group
                    if similarity_score_with_neighboring_nodes > similarity_score_with_current_nodes:
                        partition[connecting_node] = edge[0]
                    else:
                        partition[connecting_node] = edge[1]
    return partition