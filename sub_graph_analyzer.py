from collections import deque
class SubGraphAnalyzer:
    def __init__(self, subgraph):
        """
        Initialize the SubGraphAnalyzer instance with the given subgraph.

        This constructor method sets up the SubGraphAnalyzer instance with the provided subgraph and initializes an empty partition dictionary.

        Args:
            subgraph (networkx.Graph): A NetworkX subgraph representing the portion of the graph on which community detection will be performed.

        Returns:
            None

        Example:
            >>> import networkx as nx
            >>> subgraph = nx.Graph()
            >>> sga = SubGraphAnalyzer(subgraph)

        Note:
            - The `subgraph` parameter is expected to be a NetworkX graph representing the portion of the original graph on which 
                community detection will be performed.
            - After initialization, the `partition` dictionary will be used to store the community assignments for nodes in the subgraph.
        """
        

        self.graph = subgraph
        self.partition = {}

    def _bfs_with_exclusion(self, start_node, exclude_node=None, ignored_nodes=None):
        """
        Perform a breadth-first search (BFS) traversal on a graph, optionally excluding specific nodes and ignoring others.

        Args:
        - start_node (str): The node from which the BFS traversal begins.
        - exclude_node (str, optional): A node to be excluded from traversal. Defaults to None.
        - ignored_nodes (lsit, optional): A list of nodes to be ignored during traversal. Defaults to None.

        Returns:
        - list: A list containing the nodes visited during BFS traversal.

        Example:
            >>> graph = {'A': ['B', 'C'], 'B': ['A', 'D', 'E'], 'C': ['A', 'F'], 'D': ['B'], 'E': ['B', 'F'], 'F': ['C', 'E']}
            >>> analyzer = SubGraphAnalyzer(graph)
            >>> result = analyzer.bfs_with_exclusion_and_ignore('A', exclude_node='C', ignored_nodes=['B', 'F'])
            >>> print(result)
            ['A', 'D', 'E']
        
        Note:
            - This method is intended for internal use within the class and may not be directly accessible from outside the class.
        """
        
        visited = set()
        queue = deque([start_node])
        result = []

        while queue:
            node = queue.popleft()
            if node in visited:
                continue

            # Check if the current node should be excluded or ignored
            if node == exclude_node or (ignored_nodes and node in ignored_nodes):
                continue

            visited.add(node)
            result.append(node)

            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited:
                    queue.append(neighbor)

        return result
