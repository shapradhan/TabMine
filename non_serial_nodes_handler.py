from utils.embeddings import calculate_similarity_between_embeddings
from utils.graph import initialize_graph, draw_simple_graph, get_conencted_components

def identify_additional_communities(G, nodes, similarity_threshold, embeddings_dict):
    """ Identify additional communities in an existing network using node embeddings.

    Args:
        G (networkx.Graph): The existing network graph.
        nodes (list): A list of nodes for which additional communities are to be identified.
        similarity_threshold (float): The similarity threshold for considering nodes as part of a community.
        embeddings_dict (dict): A dictionary containing node embeddings.

    Returns:
        list of list: A list of list containing nodes that are members of a community.
    """

    subgraph = G.subgraph(nodes)
    subgraph_edges = list(subgraph.edges())

    edges_connecting_similar_nodes = []
    edges_connecting_dissimilar_nodes = []

    node_community_index = {n: None for n in nodes}

    for edge in subgraph_edges:
        parent_node = edge[0]
        child_node = edge[1]

        similarity_score = calculate_similarity_between_embeddings(embeddings_dict[parent_node], embeddings_dict[child_node])

        edges_connecting_similar_nodes.append(edge) if similarity_score >= similarity_threshold else edges_connecting_dissimilar_nodes.append(edge)

    new_communities = []

    for edge in edges_connecting_similar_nodes:
        parent_node = edge[0]
        child_node = edge[1]


        if len(new_communities) == 0:
            new_communities.append({parent_node, child_node})
            node_community_index[parent_node] = 0
            node_community_index[child_node] = 0
        else:
            if node_community_index[parent_node] != None:
                new_communities[node_community_index[parent_node]].add(parent_node)
                new_communities[node_community_index[parent_node]].add(child_node)
                node_community_index[child_node] = node_community_index[parent_node]

            if node_community_index[child_node] != None:
                new_communities[node_community_index[child_node]].add(child_node)
                new_communities[node_community_index[child_node]].add(parent_node)
                node_community_index[parent_node] = node_community_index[child_node]

            if node_community_index[parent_node] == None and node_community_index[child_node] == None:
                new_communities.append({parent_node, child_node})
                node_community_index[parent_node] = len(new_communities) - 1
                node_community_index[child_node] = len(new_communities) - 1
        
    for edge in edges_connecting_dissimilar_nodes:
        parent_node = edge[0]
        child_node = edge[1]

        if node_community_index[parent_node] == None:
            new_communities.append({parent_node})
            node_community_index[parent_node] = len(new_communities) - 1

    return new_communities
