from utils.graph import group_nodes_by_community
from utils.general import find_most_common_words

def labeler(partition, descriptions):
    """ Assign a text label to the communities
    
    Args:
        partition (dict): A dictionary where keys are nodes and values are the corresponding community IDs.
        descriptions (dict: A dictionary where keys are nodes and values are the corresponding descriptions.
    
    Returns:
        dict: A dictionary where keys are community IDs and the values are the corresponding labels.
    """
    
    nodes_by_communities = group_nodes_by_community(partition)
    
    community_labels = {}
    for community_id, nodes in nodes_by_communities.items():
        descriptions_by_nodes = [descriptions[node] for node in nodes]
        most_common_words = find_most_common_words(descriptions_by_nodes)
        community_labels[community_id] = ' '.join(most_common_words)
        
    return community_labels

