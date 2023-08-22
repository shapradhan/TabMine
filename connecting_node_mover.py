from utils.embeddings import calculate_average_similarity, calculate_similarity_between_embeddings, compute_average_embedding
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


def check_similarity_and_move(G, community_average_similarity_dict, connector_node, neighboring_community_id, current_community_id, similarity_threshold, partition, embeddings_dict):
    """ Check similarity between nodes and move them in appropriate community

    Args;
        G (networkx.Graph): The graph to be drawn.
        community_average_similarity_dict (dict): A dictionary where the are the community ID of neighboring communities and valeus are the average similarity score.
        connector_node (str): The connector node.
        neighboring_community_id (int): The ID of the neighboring community.
        current_community_id (int): The ID of the current community.
        similarity_threshold (float): The similarity threshold value.
        partition (dict): A dictionary where keys are nodes and values are the corresponding community IDs. 
            The communities are numbered from 0 to number of communities.
        embeddings_dict (dict): A dictionary where the keys are the table (node) names and values the embeddings of the description.
    
    Returns:
        dict: A dictionary where keys are nodes and values are the corresponding modified community IDs. 
            The communities are numbered from 0 to number of communities.
    """
    
    if community_average_similarity_dict[neighboring_community_id] >= similarity_threshold:
        # Average similarity of the community with id = neighboring_community_id is higher than the similarity threshold.

        # Get the neighboring nodes group and current nodes group of the connecting node
        neighboring_community_nodes = get_nodes_in_community(partition, neighboring_community_id)
        current_community_nodes_original = get_nodes_in_community(partition, current_community_id)

        # Current node group without the connecting node
        current_community_nodes_modified = current_community_nodes_original[:]
        current_community_nodes_modified.remove(connector_node)

        current_community_modified_embeddings = [embeddings_dict[node] for node in current_community_nodes_modified]
        average_similarity_of_current_community_modified = calculate_average_similarity(current_community_modified_embeddings)

        # Compute the similarity score between the connecting node and its neighboring node group and its own node group without itself
        similarity_score_with_neighboring_nodes = compute_similarity_between_node_and_node_group(connector_node, neighboring_community_nodes, embeddings_dict)

        similarity_score_with_current_nodes = 0
        if average_similarity_of_current_community_modified >= similarity_threshold:
            similarity_score_with_current_nodes = compute_similarity_between_node_and_node_group(connector_node, current_community_nodes_modified, embeddings_dict)
        else:
            neighboring_nodes_in_current_community = find_neighboring_nodes_in_same_community(partition, connector_node)

            for neighboring_node in neighboring_nodes_in_current_community:
                similarity_score = calculate_similarity_between_embeddings(embeddings_dict[connector_node], embeddings_dict[neighboring_node])
                if similarity_score >= similarity_score_with_current_nodes:
                    similarity_score_with_current_nodes = similarity_score
        
        # If the similarity score of the conencting node is higher with the neighboring node, move the node to that group
        # Else keep the node to the current group
        if similarity_score_with_neighboring_nodes > similarity_score_with_current_nodes:
            partition[connector_node] = neighboring_community_id
        else:
            partition[connector_node] = current_community_id
        
    else:
        # Average similarity of the community with id = neighboring_community_id is lower than the similarity threshold.
        
        neighboring_nodes_dict = {}
        neighboring_nodes = list(G.neighbors(connector_node))
        
        for neighbor in neighboring_nodes:
            neighbor_community = partition[neighbor]
            if neighbor_community in neighboring_nodes_dict:
                neighboring_nodes_dict[neighbor_community].append(neighbor)
            else:
                neighboring_nodes_dict[neighbor_community] = [neighbor]

        highest_similarity_score = 0
        community_to_move_to = None
        for neighboring_community_id, neighboring_node in neighboring_nodes_dict.items():
            sim_score = calculate_similarity_between_embeddings(embeddings_dict[connector_node], embeddings_dict[neighboring_node[0]])
            if sim_score >= highest_similarity_score:
                highest_similarity_score = sim_score
                community_to_move_to = neighboring_community_id

        partition[connector_node] = community_to_move_to

    return partition


def move_connecting_nodes(partition, nodes_by_community, community_connecting_nodes_dict, embeddings_dict, similarity_threshold, G):
    """ Move connecting nodes to a more similar community 

    Args:
        partition (dict): A dictionary where keys are nodes and values are the corresponding community IDs. 
            The communities are numbered from 0 to number of communities.
        nodes_by_community (dict): A dictionary where keys are the community IDs and values are the list of nodes in respective communities.
        community_connecting_nodes_dict (dict): A dictionary where keys are tuples connecting two values representing neighboring 
            communities. The values are the set of nodes connecting the two communities represented by the tuple. The set contains
            either 0 or 1 item.
        embeddings_dict (dict): A dictionary with table (node) name as the key and the embeddings of its descriptions as the value.
        similarity_threshold (float): A similarity threshold value.
        G (networkx.Graph): The graph to be drawn.

    Returns:
        dict: A dictionary where keys are connecting nodes and values are the ID of the community in which it belongs to after
            they have been moved, if necessary.
    """

    community_average_similarity_dict = {}
    for neighboring_community_id, community_nodes in nodes_by_community.items():
        community_embeddings = [embeddings_dict[node] for node in community_nodes]
        average_similarity_of_communities = calculate_average_similarity(community_embeddings)
        community_average_similarity_dict[neighboring_community_id] = average_similarity_of_communities 

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
        # Nodes connect more than two communities
        for connector_node in set(nodes_connecting_multiple_communities):
            neighboring_community_nodes = {}
            current_community_id = None

            for edge, nodes in community_connecting_nodes_dict.items():
                if nodes:
                    nodes = list(nodes)
                    if connector_node in nodes:
                        current_community_id = edge[1]
                        neighboring_community_id = edge[0]
                        nodes_already_checked.append(connector_node)
                        neighboring_community_nodes[neighboring_community_id] = get_nodes_in_community(partition, neighboring_community_id)
                       

            for neighboring_community_id in neighboring_community_nodes.keys():
                partition = check_similarity_and_move(
                    G,
                    community_average_similarity_dict, 
                    connector_node, 
                    neighboring_community_id, 
                    current_community_id, 
                    similarity_threshold, 
                    partition, 
                    embeddings_dict)

    else:
        # Nodes connect at most two communities
        for edge, nodes in community_connecting_nodes_dict.items():
            if nodes:
                nodes = list(nodes)
                neighboring_community_nodes = {}

                for connector_node in nodes:
                    neighboring_community_id = edge[0]
                    current_community_id = edge[1]

                    partition = check_similarity_and_move(
                        G,
                        community_average_similarity_dict, 
                        connector_node, 
                        neighboring_community_id, 
                        current_community_id, 
                        similarity_threshold, 
                        partition, 
                        embeddings_dict)
                    
    return partition