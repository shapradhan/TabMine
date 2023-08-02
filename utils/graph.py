import networkx as nx

def initialize_graph(nodes, edges, directed=False):
    """Intialize a Networkx graph

    Args:
        nodes (list): A list of nodes representating the tables in a database.
        edges (list): A list of tuples representing edges which represent the foreign key relationship between two nodes.
                    Each tuple should contain two node identifiers.A list of edges.
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
    """
    Get nodes belonging to a specific community.

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
    """

    community_nodes = {}
    for node, community_id in partition.items():
        # Get the community_id and make that a key of an empty list, in which the nodes are appended
        community_nodes.setdefault(community_id, []).append(node)
    return community_nodes

def find_connecting_nodes(graph, community_nodes):
    """
    Find nodes that connect a community to nodes outside the community.

    Parameters:
        graph (networkx.Graph): The graph in which to search for connecting nodes.
        community_nodes (list): A list of nodes representing a community within the graph.

    Returns:
        list: A list of tuples representing the nodes that connect the specified community to nodes outside the community.
    """

    connecting_nodes = {}
    for c1 in community_nodes:
        for c2 in community_nodes:
            if c1 != c2:
                connecting_nodes[(c1, c2)] = set()
                for node in community_nodes[c1]:
                    neighbors = graph.neighbors(node)
                    intersecting_node = set(neighbors).intersection(community_nodes[c2])
                    if intersecting_node:
                        connecting_nodes[(c1, c2)].update(intersecting_node)
    return connecting_nodes