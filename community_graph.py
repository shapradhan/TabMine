import networkx as nx

class Graph(nx.Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_connected_components(self, nodes_in_community):
        subgraph = self.subgraph(nodes_in_community)
        return list(nx.connected_components(subgraph))