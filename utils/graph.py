import matplotlib.pyplot as plt
import networkx as nx

from utils.general import get_word_between_strings

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
    """

    community_nodes = {}
    for node, community_id in partition.items():
        # Get the community_id and make that a key of an empty list, in which the nodes are appended
        community_nodes.setdefault(community_id, []).append(node)
    return community_nodes

def find_connecting_nodes(graph, nodes_by_community):
    """Find nodes that connect a community to nodes outside the community.

    Parameters:
        graph (networkx.Graph): The graph in which to search for connecting nodes.
        nodes_by_community (list): A list of nodes representing a community within the graph.

    Returns:
        list: A list of tuples representing the nodes that connect the specified community to nodes outside the community.
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

def draw_graph(G, partition, title):
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

    for node in G.nodes():
        community_id = partition[node]
        node_color = plt.cm.tab20(community_id)
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=node_color, node_size=200, label=f"Community {community_id}")

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

    plt.title(title)
    plt.legend()
    plt.axis('off')
    plt.show()

def _insert_into_list(lst, target_value, new_value, position):
    """Insert a new value into the given list before or after a specified target value, depending on the value of position.

    Args:
        lst (list): The list in which the new value will be inserted.
        target_value (str): The value after which the new value should be inserted.
        new_value (str): The value to be inserted into the list.
        position (str): The position where the new value should be inserted. 
            It has two possible values 'before' the target value or 'after the target value.

    Returns:
        list: A modified list with the new value inserted before or after the target value.
    """
    try:
        index = lst.index(target_value)
        if position == 'before':
            lst.insert(index, new_value)
        elif position == 'after':
            lst.insert(index + 1, new_value)
        else:
            raise ValueError('Proper position is not provided.')
    except ValueError:
        print("Target value not found in the list.")
    
    return lst

def _check_value_in_lists(value, list1, list2):
    """Find if a value is in either of the two lists.

    This function takes a value and two lists as parameters, and it searches for the value in both lists. 
    If the value is found in the first list, 0 is returned. 
    If the value is found in the second list, 1 is returned.
    If the value is not found in either list, -1 is returned. 

    Args:
        value (any): The value to search for in the lists.
        list1 (list): The first list to search in.
        list2 (list): The second list to search in.

    Returns:
        int: A value representing which list is the given value is found or -1 if the value is not found in either list.
    """
    
    return 0 if value in list1 else (1 if value in list2 else -1)
