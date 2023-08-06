from utils.embeddings import calculate_similarity_between_embeddings


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


def detect_communities_in_node_series(nodes, embeddings_dict, similarity_threshold):
    """ Detect communities in a series chain of nodes.

    This function can be used after connecting nodes have been moved after the initial partion was identified.
    Although, it is not stricly necessary for this function operate.
    However, the general idea of this function is that it is expected to be run after the connecting nodes have been moved, if necessary.

    Args:
        nodes (list): A list of nodes in which communities have to be detected.
        similarity_threshold (float): The similarity threshold value.
        embeddings_dict (dict): A dictionary with table (node) name as the key and the embeddings of its descriptions as the value.
    
    Returns:
        list: A list of list with nodes partitioned into communities, with each inner list being a community.
    """

    communities = []
    if len(nodes) > 1:
        for i in range(0, len(nodes) - 1):
            node1 = nodes[i]
            node2 = nodes[i + 1]

            above_similarity_threshold = compare_average_similarity_and_threshold(node1, node2, similarity_threshold, embeddings_dict)
            
            if above_similarity_threshold:
                # Similarity between node1 and node2 is higher than or equal to the similarity threshold.
                if len(communities) == 0:
                    # First run. So, add both node1 and node2 in the communities list.
                    # They are added with them being together inside a list because the similarity threshold between them
                    # is higher than or equal to the similarity threshold.
                    communities.extend([[node1, node2]])
                else:
                    for community in communities:
                        # If node1 is already there in a community and its similarity with node2 
                        # is higher than or equal to the similarity threshold, then add node2 in the same community.
                        if node1 in community:
                            community.append(node2)
            else:
                # Similarity between node1 and node2 is lower than the similarity threshold.
                if len(communities) == 0:
                    # First run. So, add both node1 and node2 in the communities list.
                    # They are added in separate inner lists because the similarity threshold between them
                    # is lower than the similarity threshold.
                    communities.extend([[node1], [node2]])
                else:
                    # If node1 is already there in a community (denoted by communities[-1]) but its similarity 
                    # with node2 is lower than the similarity threshold, then add node2 in a different community.
                    if node1 in communities[-1]:
                        communities.append([node2])       
    else:
        # Length of communities less than or equal to 1
        # The given nodes are the communities
        communities.append(nodes)
    
    return communities