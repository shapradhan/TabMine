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
