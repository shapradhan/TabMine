class Community:
    def __init__(self, partition):
        """
        Initialize the Community instance with the given partition.

        This constructor method sets up the Community instance with the provided partition, along
        with initializing dictionaries for storing nodes by community and neighbor count by connector nodes.

        Args:
            partition (dict): A dictionary where keys are nodes and values are the corresponding community IDs.

        Returns:
            None

        Example:
            >>> partition = {0: 1, 1: 1, 2: 2, 3: 2}
            >>> cs = Community(partition)

        Note:
            - After initialization, the `nodes_by_community` dictionary will store nodes grouped by their respective communities, 
                and the `neighbor_count_by_connector_nodes` dictionary will store the neighbor count for connector nodes.
        """
        
        self.partition = partition
        self.nodes_by_community = {}
        self.neighbor_count_by_connector_nodes = {}
                neighbors = list(graph.neighbors(node))
                
                for neighbor in neighbors:
                    embeddings_to_check = [embeddings_dict[node], embeddings_dict[neighbor]]
                    avg_similarity_score = calculate_average_similarity(embeddings_to_check)
        
                    if avg_similarity_score > highest_similarity_score:
                        highest_similarity_score = avg_similarity_score
                        community_to_move_to = partition[neighbor]

                partition = self._move(graph, node, node_original_community_id, community_to_move_to, max_community_id)
            
            else:
                neighboring_communities = self._get_neighboring_communities(graph, node) # Includes current community of the node as well
               
                for community_id in neighboring_communities: 
                    nodes_in_neghboring_community = self._get_nodes_by_community_id(community_id)
                    if node not in nodes_in_neghboring_community:
                        nodes_in_neghboring_community.append(node)   # Add the node to check if adding it in the neighboring communities makes the score go higher

                    embeddings = [embeddings_dict[node] for node in nodes_in_neghboring_community]
                    avg_similarity_score = calculate_average_similarity(embeddings)
                    
                    if avg_similarity_score > highest_similarity_score:
                        highest_similarity_score = avg_similarity_score
                        community_to_move_to = community_id

                partition = self._move(graph, node, node_original_community_id, community_to_move_to, max_community_id)
        return partition

    def get_neighbor_count_of_connector_nodes(self, graph, reverse=False):
        """
        Counts the number of neighboring nodes of the connector nodes in a graph

        Args:
            graph (networkx.Graph): The graph representing the relationships between the tables.
            reverse (bool): A Boolean value indicating whether the output should be in ascending or descending order. 
                True indicates ascending and False indicates descending order.
            
        Returns:
            dict: A dictionary containing the count of the number of neigboring nodes where the keys represent the connector nodes 
                and the values represent the count of the neighboring communities.
        """
        partition = self.partition
    
        neighbor_count = {}
        
        for node1, node2 in graph.edges():
            if node1 not in neighbor_count or node2 not in neighbor_count:
                if partition[node1] != partition[node2]:  # Nodes in different communities
                    neighbor_count[node1] = len(list(graph.neighbors(node1)))
                    neighbor_count[node2] = len(list(graph.neighbors(node2)))

        self.neighbor_count_by_connector_nodes = dict(sorted(neighbor_count.items(), key=lambda item: item[1], reverse = reverse))
        return self.neighbor_count_by_connector_nodes