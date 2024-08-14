import matplotlib.pyplot as plt
import networkx as nx
import os

class Graph(nx.Graph):
    """
    A custom Graph class that extends the NetworkX Graph class.

    This class inherits from `networkx.Graph` and allows to create and manipulate 
    undirected graphs with optional additional functionality.

    Args:
        *args (tuple, optional): Positional arguments passed to the `networkx.Graph` constructor.
        **kwargs (dict, optional): Keyword arguments passed to the `networkx.Graph` constructor.

    Attributes:
        All attributes of `networkx.Graph` are available.

    Methods:
        All methods of `networkx.Graph` are available.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_connected_components(self, nodes_in_community):
        """
        Identify and return the connected components within a given set of nodes in a community.

        This method finds all the connected subgraphs (components) within the specified subset of nodes.
        Each connected component is a maximal subgraph in which any two vertices are connected to each other
        by paths, and which is connected to no additional vertices in the containing graph.

        Args:
            nodes_in_community (list): A list of nodes representing a community within the graph. 

        Returns:
            List[set]: A list of sets, where each set contains the nodes that form a connected component within the
            specified community. Each connected component is represented as a set of nodes.

        Example:
            >>> nodes_in_community = [1, 2, 3, 4, 5]
            >>> connected_components = self.get_connected_components(nodes_in_community)
            >>> print(connected_components)
            [{1, 2}, {3, 4, 5}]
        """

        subgraph = self.subgraph(nodes_in_community)
        return list(nx.connected_components(subgraph))

    def display_graph(self, partition, save=False, save_dir_path=None):
        """
        Visualize the graph with nodes colored based on the given partition.

        This method displays the graph, coloring the nodes according to the specified partition, which groups nodes into different 
        communities or clusters. Optionally, the visualization can be saved
        to a file.

        Args:
        partition (dict): A dictionary where keys are node identifiers and values are integers representing the community or each node belongs to.
        save (bool, optional): A flag indicating whether to save the visualization to a file. If True, the graph visualization will be saved to a file. 
            The default is False.
        save_dir_path(str, optional): The path of the directory in which the visualization should be saved.

        Returns:
            None

        Example:
            >>> partition = {0: 1, 1: 1, 2: 2, 3: 2}
            >>> self.display_graph(partition, save=True)
            
        Note:
            - If `save` is True, the visualization will be saved to a file named 'graph.png' in the desktop.
        """

        pos = nx.spring_layout(self)  # Position nodes using the spring layout algorithm
        node_colors = [partition[node] for node in self.nodes()]
        nx.draw(self, pos, with_labels=True, node_color=node_colors, cmap=plt.cm.tab10, node_size=1500, font_size=12, font_weight='bold')
        nx.draw_networkx_edge_labels(self, pos)
        
        # Get the path to the desktop directory
        # save_dir_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

        # Set the figure size to fill the entire screen
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)  

        if save:
            file_path = os.path.join(save_dir_path, "graph.png")
            plt.savefig(file_path)

            os.startfile(file_path)
        else:
            plt.show()
    
    def convert_igraph_to_networkx_graph(self, G_igraph, partition):
        """
        Converts an igraph graph into a NetworkX graph, preserving community information.

        This method takes an igraph graph `G_igraph` and converts it into a NetworkX graph,
        adding nodes and edges accordingly. Each node in the NetworkX graph is annotated 
        with a `community` attribute based on the provided `partition` dictionary.

        Args:
            G_igraph (igraph.Graph): The igraph graph that is to be converted into a NetworkX graph.
            partition (dict):A dictionary where keys are node names from `G_igraph` and values are community 
                identifiers. This information is added as a `community` attribute to each node in the NetworkX graph.

        Returns:
            None: This method modifies the current NetworkX graph in place by adding nodes and edges from `G_igraph`.
        
        Notes:
            - The method assumes that nodes in `G_igraph` have a 'name' attribute that is used as their identifier.
            - The edges are added based on the connections in `G_igraph` without any additional attributes.
        """
        
        # Add nodes with community membership as an attribute
        for node in G_igraph.vs:
            # G_nx.add_node(node['name'], community=partition[node['name']])
            self.add_node(node['name'], community=partition[node['name']])

        # Add edges to the networkx graph
        for edge in G_igraph.es:
            source, target = edge.tuple
            self.add_edge(G_igraph.vs[source]['name'], G_igraph.vs[target]['name'])
            