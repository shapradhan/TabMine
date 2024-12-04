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

    def wrap_label(self, label, max_length):
        """
        Wraps a label text into multiple lines with a specified maximum line length.

        Args:
            label (str): The label text to be wrapped.
            max_length (int): The maximum length of each line.

        Returns:
            str: The wrapped label text with each line not exceeding the specified maximum length.
        
        Example:
            label = "This is a long label that needs to be wrapped."
            max_length = 10
            wrapped_label = wrap_label(label, max_length)
            print(wrapped_label)
        """
        words = label.split()
        wrapped_label = []
        current_line = []

        for word in words:
            # Check if adding the word exceeds the maximum line length
            if sum(len(w) for w in current_line) + len(word) + len(current_line) > max_length:
                wrapped_label.append(' '.join(current_line))
                current_line = [word]
            else:
                current_line.append(word)

        # Add any remaining words in current_line as the last line
        if current_line:
            wrapped_label.append(' '.join(current_line))

        return '\n'.join(wrapped_label)


    def display_graph(self, partition, save=False, filename='graph.png', save_dir_path=None):
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

        plt.clf()  # Clear the current figure
        pos = nx.spring_layout(self)  # Position nodes using the spring layout algorithm

        # Create a list of colors based on the partition
        node_colors = [partition[node] for node in self.nodes()]

        # Create labels that include community IDs
        labels = {node: f"{node} (Community {partition[node]})" for node in self.nodes()}

        max_length = 15  # Set your desired max length
        labels = {node: self.wrap_label(f"{node} (C {partition[node]})", max_length) for node in self.nodes()}

        # Draw the graph with the updated labels
        nx.draw(self, pos, with_labels=True, labels=labels, node_color=node_colors, cmap=plt.cm.tab10, node_size=2000, font_size=20, font_weight='bold')
        nx.draw_networkx_edge_labels(self, pos)
                
        # Set the figure size to fill the entire screen
        fig = plt.gcf()
        fig.set_size_inches(50, 30)  

        if save:
            file_path = os.path.join(save_dir_path, filename)
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
            self.add_node(node['name'], community=partition[node['name']])

        # Add edges to the networkx graph
        for edge in G_igraph.es:
            source, target = edge.tuple
            self.add_edge(G_igraph.vs[source]['name'], G_igraph.vs[target]['name'])