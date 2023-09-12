import matplotlib.pyplot as plt
import networkx as nx

from collections import Counter

from utils.general import get_word_between_strings, check_value_in_list
from utils.embeddings import calculate_similarity_between_embeddings


def initialize_graph(nodes, edges, directed=False):
    """Intialize a Networkx graph

    Args:
        nodes (list): A list of nodes representating the tables in a database.
        edges (list): A list of tuples where each tuple represents an edge between two nodes.
            Each tuple should contain exactly two elements - the source node and the target node. 
        directed (bool):  If True, the graph is directed; if False, the graph is undirected.
                          Defaults to False.

    Returns:
        networkx.DiGraph or networkx.Graph: An initialized NetworkX graph object.
                                            If directed is True, a DiGraph is returned; otherwise, a Graph is returned.
    """   

    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def get_nodes_in_community(partition, community_id):
    """Get nodes belonging to a specific community.

    Args:
        partition (dict): A dictionary where keys are nodes and values are community IDs.
        community_id (int): The ID of the community to extract.

    Returns:
        list: A list of nodes belonging to the specified community.
    """

    return [node for node, comm_id in partition.items() if comm_id == community_id]


def group_nodes_by_community(partition):
    """Group nodes based on the community partition.

    Args:
        partition (dict): A dictionary where keys are nodes and values are the corresponding community IDs.

    Returns:
        dict: A dictionary where keys are community IDs and values are lists of nodes belonging to each community.

    Example:
        partition = {
            'vbrk': 0, 'bkpf': 0, 'likp': 1, 'lips': 1, 'bseg': 0, 'vbak': 3, 
            'vbap': 1, 'cdhdr': 2, 'nast': 3, 'cdpos': 2, 'vbfa': 1, 'vbrp': 0
        }

        Returns {
            0: ['vbrk', 'bkpf', 'bseg', 'vbrp'], 
            1: ['likp', 'lips', 'vbap', 'vbfa'], 
            2: ['cdhdr', 'cdpos'],
            3: ['vbak', 'nast']
        }
    """

    community_nodes = {}
    for node, community_id in partition.items():
        # Get the community_id and make that a key of an empty list, in which the nodes are appended
        community_nodes.setdefault(community_id, []).append(node)
    return community_nodes


def find_communities_connecting_nodes(graph, nodes_by_community):
    """Find nodes that connect a community to nodes outside the community.

    Parameters:
        graph (networkx.Graph): The graph in which to search for connecting nodes.
        nodes_by_community (list): A list of nodes representing a community within the graph.

    Returns:
        dict: A dictionary in which the keys are the tuples representing the communities that are 
            connected by the nodes, which are the values in the dictionary.

    Example:
        nodes_by_community =  {
            0: ['vbrk', 'bkpf', 'bseg', 'vbrp'], 
            1: ['likp', 'lips', 'vbap', 'vbfa'], 
            2: ['cdhdr', 'cdpos'],
            3: ['vbak', 'nast']
        }

        Returns {
            (0, 1): {'vbfa'}, (0, 3): set(), (0, 2): set(), (1, 0): {'vbrp'}, 
            (1, 3): {'vbak'}, (1, 2): set(), (3, 0): set(), (3, 1): {'vbap'}, 
            (3, 2): set(), (2, 0): set(), (2, 1): set(), (2, 3): set()
        }
            Here, communities with IDs 0 and 1 are connected by the node 'vbfa.' 
            Communities 0 and 3 are not connected with each other.
    """

    connecting_nodes = {}
    for c1 in nodes_by_community:
        for c2 in nodes_by_community:
            if c1 != c2:
                connecting_nodes[(c1, c2)] = set()
                for node in nodes_by_community[c1]:
                    neighbors = graph.neighbors(node)
                    intersecting_node = set(neighbors).intersection(nodes_by_community[c2])
                    if intersecting_node:
                        connecting_nodes[(c1, c2)].update(intersecting_node)
    return connecting_nodes


def _get_parent_node(foreign_key_relation_list):
    """ Extract the name of the parent node 
    
    Parent node is the table whose primary key is used as the foreign key in another table.
    The foreign key relation text is the third item, denoted by index 2, in the list.
    The name is after the word 'REFERENCES' and before the opening parenthesis. 
    
    Args:
        foreign_key_relation_list (list): A list containing the foreign key relation text, among others.
    
    Returns:
        str: The name of the parent node (table)
    """

    start_str = 'REFERENCES'
    end_str = '('
    return get_word_between_strings(foreign_key_relation_list[2], start_str, end_str).strip()


def get_edges(foreign_key_relation_list):
    """ Get the nodes representing the two ends of an edge

    The first item of the tuple is the parent node (i.e., the table whose primary key is used as the foreign key in another table).
    The second item of the tuple is the child node (i.e., the table that is using the primary key of another table as the foreign key).

    Args:
        foreign_key_relation_list (list): A list containing the foreign key relation text, among others.

    Returns:
        tuple: A tuple representing the nodes at the two ends of an edge    
    """

    child_node = foreign_key_relation_list[0]
    parent_node = _get_parent_node(foreign_key_relation_list)
    return (parent_node, child_node)


def draw_graph(G, partition, title, labels=None, color_map=None):
    """
    Draws and displays a graph with nodes colored based on the provided partition.

    This function uses the matplotlib library to visualize the graph G. Nodes are colored
    according to the given partition, which is a dictionary where keys are node IDs and
    values are integers representing the partition index. The title parameter is used to
    set the title of the graph plot.

    Parameters:
    G (networkx.Graph): The graph to be drawn.
    partition (dict): A partition of the graph nodes, where keys are node IDs and values
                     are integers representing the partition index.
    title (str): The title to be displayed above the graph plot.

    Returns:
        None
    """
     
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 6))

    node_colors = [partition[node] for node in G.nodes()]

    node_labels = {node: str(node) for node in G.nodes()} 

    # Use color map - Options e.g., Pastel1, tab20c, viridis
    color_map = plt.cm.get_cmap(color_map, max(node_colors) + 1)
    node_colors_rgb = [color_map(color) for color in node_colors]

    # Draw the graph components
    nx.draw_networkx_nodes(G, pos, node_color=node_colors_rgb, node_size=300)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color='black')

    # Create a custom legend using the same colors as node colors
    legend_labels = {community_id: label for community_id, label in set(labels.items())} if labels else {community: f"Community {community}" for community in set(partition.values())}
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markersize=10,
                                markerfacecolor=color_map(community))
                    for community, label in legend_labels.items()]

    plt.legend(handles=legend_handles)

    plt.title(title)
    plt.axis('off')
    plt.show()


def get_relevant_edges(community_node_group, all_edges):
    """ Get relevant edges from the full edge list based on the nodes in a community

    Args:
        community_node_group (list): A list of nodes in a commmunity
        all_edges (list): A list of tuples where each tuple represents an edge between two nodes.
            Each tuple should contain exactly two elements - the source node and the target node. 

    Returns:
        list: A list of tuples representing the source and the target nodes of the edges.
            The source and the target nodes are both must be in the same community.

    Example:
        community_node_group = ['vbrk', 'bkpf', 'bseg', 'vbrp']
        Returns [('vbrk', 'bkpf'), ('bkpf', 'bseg'), ('vbrk', 'vbrp')]
    """

    relevant_edges = []
    for i in all_edges:
        source_node = i[0]
        target_node = i[1]

        if source_node in community_node_group and target_node in community_node_group:
            relevant_edges.append(i)
    return relevant_edges


def _get_non_common_nodes(nodes, source_nodes, target_nodes):
    """ Get the nodes that are only in the source nodes but not in the target nodes.

    Args:
        nodes (list): List of all nodes.
        source_nodes (list): List of source nodes.
        target_nodes (list): List of target nodes.

    Returns:
        list: A list containing the nodes that are only in the source nodes but not in the target nodes.
    """
    
    return [node for node in nodes if node in source_nodes and node not in target_nodes]


def find_central_node(nodes, source_nodes, target_nodes, graph, centrality_measure='betweenness'):
    """ Identify the node that could be the central node in a given list of nodes.
    
    Given a list of nodes, check whether the nodes in the list are both source node and target node of edges.
    The central node is that node, which is only found in the list of source nodes as it is the main table without any foreign key.

    Args:
        nodes (list): A list of all nodes.
        source_nodes (list): A list of source nodes.
        target_nodes (list): A list of target nodes.
        graph (networkx.DiGraph or networkx.Graph): The graph in which centrality is calculated
        centrality_measure (str, optional): The measure of centrality to be used. Defaults to be the betweenness centrality measure.

    Returns:
        str: The node that is the central among the nodes in a given list of nodes.
    """
    
    # Identify the nodes that are only in the source nodes but not in the target nodes.
    common_nodes = _get_non_common_nodes(nodes, source_nodes, target_nodes)

    # If there is only one non common node, then that is the central node.
    # Otherwise, identify the central node using the centrality measure.
    if len(common_nodes) == 1:
        return common_nodes[0]
    else:
        if centrality_measure == 'degree':
            centrality = nx.degree_centrality(graph)
        elif centrality_measure == 'betweenness':
            centrality = nx.betweenness_centrality(graph)
        elif centrality_measure == 'eigenvector':
            centrality = nx.eigenvector_centrality(graph)

        central_node = max(centrality, key=centrality.get)
        return central_node
  

def arrange_nodes_in_series(edges):
    """ Get the series chain from given list of tuples representing the edge between two nodes.

    This function takes a list of edges and arranges the nodes in an appropriate order
    to maintain the sequence defined by the edges. Each edge in the list is a tuple
    representing a connection between two nodes. The ordering depends on the edges 
    between the nodes.

    Args:
        edges (list of tuple): A list of tuples where each tuple represents an edge between two nodes. 
            Each tuple should contain exactly two elements - the source node and the target node. 

    Returns:
        list: A list of nodes arranged in the order defined by the given edges.

    Example:
        edges = [('vbak', 'nast'), ('vbak', 'vbap'), ('vbfa', 'vbap')]
        Returns ['vbfa', 'vbap', 'vbak', 'nast']
        Here, vbak is the parent node of nast and vbap nodes. vbfa is the parent of vbap node.
        So, vbak -> nast, vbak -> vbap, vbfa -> vbap.
        When written in a series: vbfa -> vbap <- vbak -> nast. Therefore, that is the the return series.
    """
    
    # Identify the unique nodes from the edges
    unique_nodes = set(node for tupl in edges for node in tupl)
    temp_list = []
    final_list = []

    # Loop until the final list has all the unique nodes
    while len(final_list) != len(unique_nodes):
        for tupl in edges:
            print('checking tuple {0}'.format(tupl))
            source_node = tupl[0]
            target_node = tupl[1]

            # Check whether the source node or the target node is in the final list
            existing_node_in_final_list = check_value_in_list(source_node, target_node, final_list)

            # Identify the node of the tuple that is not already in the final list
            node_not_in_final_list = target_node if existing_node_in_final_list == source_node else source_node if existing_node_in_final_list == target_node else None
    
            # In the first run, add the source node and target node to the final list already as the final list is empty
            if len(final_list) == 0:
                final_list.extend([source_node, target_node])

            if existing_node_in_final_list:
                existing_node_index = final_list.index(existing_node_in_final_list)

                # If existing node is not in the first position in the final list, then append the node that is still not in the final list to the end of the list
                # E.g., if existing node = b, another node = c and final list = [a, b], then the final list becomes [a, b, c]
                # If existing node is in the first position in the final list, then insert the node that is still not in the final list to the start of the list
                # E.g., if existing node = a, another node = c and final list = [a, b], then the final list becomes [c, a, b]
                if existing_node_index > 0:
                    final_list.append(node_not_in_final_list)
                elif existing_node_index == 0:
                    final_list.insert(0, node_not_in_final_list)

            # If both source node and target node are not in the final list, then add the tuple to a temporary list for another pass through the while-loop
            if source_node not in final_list or target_node not in final_list:
                temp_list.append(tupl)
                
            # Change edges to temp_list to terminate the while-loop after a certain number of iterations
            edges = temp_list
    return final_list


def convert_communities_list_to_partition(communities):
    """ Convert a list of communities into a dictionary representing the partitions.

    Args:
        communities (list): A list of list with nodes partitioned into communities, with each inner list being a community.

    Returns: 
        dict: The partition, with communities numbered from 0 to number of communities
    """

    partition = {}

    for community in communities:
        index = communities.index(community)
        for node in community:
            partition[node] = index

    return partition

def check_any_node_more_than_two_outgoing_edges(edges, nodes_by_community):
    """ Check if any of the nodes have more than two outgoing edges.

    Args:
        edges (list): A list of tuples representing the nodes connected by the edges.
        nodes_by_community (dict): A dictionary where the keys are the community ID and values are the list of nodes in that community.

    Returns:
        bool: True if any node has more than two outgoing edges; otherwise False.
    """

    for nodes in nodes_by_community.values():
        relevant_edges = get_relevant_edges(nodes, edges)
        edge_count = Counter()

        for edge in relevant_edges:
            edge_count.update(edge)

        for count in edge_count.values():
            if count > 2:
                return True
        return False

def check_edge_connection_between_nodes(node1, node2, edges):
    """ Check if there is an edge between given nodes.

    Args:
        node1 (str): A node to check if there is an edge connection.
        node2 (str): A node to check if there is an edge connection.
        edges (list): A list of tuples representing the nodes connecting an edge.
    
    Returns:
        bool: True if there is an edge between two given nodes; otherwise False.
    """
    
    for edge in edges:
        source_node = edge[0]
        target_node = edge[1]
        if (source_node == node1 and target_node == node2) or (source_node == node2 and target_node == node1):
            return True
    return False

def find_similar_nodes(node, nodes, edges, similarity_threshold, embeddings_dict):
    """ Find similar nodes to a given node.

    Args:
        node (str): A node for which similar nodes have to be found.
        nodes (list): A list of nodes.
        edges (list): A list of tuples representing the nodes connecting an edge.
        similarity_threshold (float): The similarity threshold value.
        embeddings_dict (dict): A dictionary where the keys are the table (node) names and values the embeddings of the description.
    
    Returns:
        list, list: Two lists containing similar and dissimilar nodes for a given node
    """

    similar_nodes = []
    dissimilar_nodes = []
    
    for other_node in nodes:
        if other_node != node:
            sim_score = calculate_similarity_between_embeddings(embeddings_dict[node], embeddings_dict[other_node])
            if sim_score >= similarity_threshold:
                # Check if there is a direct connection between the two nodes via an edge.
                if check_edge_connection_between_nodes(node, other_node, edges):
                    similar_nodes.append(other_node)
                else:
                    dissimilar_nodes.append(other_node)
            else:
                dissimilar_nodes.append(other_node)
    
    return similar_nodes, dissimilar_nodes

def find_additional_communities(nodes, edges, similarity_threshold, embeddings_dict):
    """ Find additional communities recursively.

    Args:
        nodes (list): A list of nodes.
        edges (list): A list of tuples representing the nodes connecting an edge.
        similarity_threshold (float): The similarity threshold value.
        embeddings_dict (dict): A dictionary where the keys are the table (node) names and values the embeddings of the description.
    
    Returns:
        list: A list of list with nodes of the additional communities in the inner lists.
    """

    if not nodes:
        return []

    node = nodes[0]
    similar_nodes, remaining_nodes = find_similar_nodes(node, nodes, edges, similarity_threshold, embeddings_dict)

    # Recursively process remaining nodes
    recursive_results = find_additional_communities(remaining_nodes, edges, similarity_threshold, embeddings_dict)
    
    # Group the node and its similar nodes together
    group = [node] + similar_nodes
     
    return [group] + recursive_results if group else recursive_results

def draw_simple_graph(G):
    """ Draw a simple undirected graph using the networkx library.

    Args:
        G (networkx.Graph): The graph to be drawn.

    Returns:
        None
    """
    
    pos = nx.spring_layout(G) 
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=12, font_color='black', font_weight='bold')
    plt.show() 

