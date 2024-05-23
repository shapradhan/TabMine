import matplotlib.pyplot as plt
import networkx as nx
import os

class Graph(nx.Graph):
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