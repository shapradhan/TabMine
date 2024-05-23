from utils.embeddings import calculate_average_similarity

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

    def group_nodes_by_community(self):
        """
        Group nodes based on the community partition.

        Args:
            partition (dict): A dictionary where keys are nodes and values are the corresponding community IDs.

        Returns:
            dict: A dictionary where keys are community IDs and values are lists of nodes belonging to each community.

        Example:
            >>> partition = {
                'vbrk': 0, 'bkpf': 0, 'likp': 1, 'lips': 1, 'bseg': 0, 'vbak': 3, 
                'vbap': 1, 'cdhdr': 2, 'nast': 3, 'cdpos': 2, 'vbfa': 1, 'vbrp': 0
            }
            >>> c = Community(partition)
            >>> nodes_by_community = c.group_nodes_by_community()
            >>> print(nodes_by_community)
            {
                0: ['vbrk', 'bkpf', 'bseg', 'vbrp'], 
                1: ['likp', 'lips', 'vbap', 'vbfa'], 
                2: ['cdhdr', 'cdpos'],
                3: ['vbak', 'nast']
            }
        """

        for node, community_id in self.partition.items():
            # Get the community_id and make that a key of an empty list, in which the nodes are appended
            self.nodes_by_community.setdefault(community_id, []).append(node)
        return self.nodes_by_community

    def _get_nodes_by_community_id(self, community_id):
        """
        Get list of nodes given a community ID

        Args:
            community_id (int): An ID of the community for which nodes belonging to that community have to be identified.

        Returns:
            list: A list of nodes for the given community ID.

        Example:
            >>> c = Community({
                'vbrk': 0, 'bkpf': 0, 'likp': 1, 'lips': 1, 'bseg': 0, 'vbak': 3,  'vbap': 1, 'cdhdr': 2, 'nast': 3, 'cdpos': 2, 'vbfa': 1, 'vbrp': 0
            })
            >>> nodes_by_community_id = c.get_nodes_by_community_id(1)
            >>> print(nodes_by_community_id)
            ['likp', 'lips', 'vbap', 'vbfa']
        """

        return [node for node, comm_id in self.partition.items() if comm_id == community_id]
    
    def _get_neighboring_communities(self, graph, node):
        """
        Retrieves the neighboring communities of a given node in a graph.

        Args:
            graph (networkx.Graph): The graph representing the relationships between the tables.
            node (str): The node for which neighboring communities are to be found.

        Returns:
            list: A list of neighboring communities of the given node.
        """

        neighbors = graph.neighbors(node)
        community_set = set()
        
        for neighbor in neighbors:
            community_id = self.partition[neighbor]
            community_set.add(community_id)

        return list(community_set) 

    def _get_nodes_by_community_id(self, community_id):
        """
        Retrieves the nodes in the community with a given ID.

        Args:
            community_id (int): The ID of the community for which nodes have to be identified.

        Returns:
            list: A list containing the nodes in a given community
        """

        return [node for node, comm_id in self.partition.items() if comm_id == community_id]

    def _move(self, graph, node, node_original_community_id, community_to_move_to, max_community_id):
        """
        Retrieves the nodes in a given community and moves the node to a the community with community ID = community_to_move_to

        Args:
            graph (networkx.Graph): The graph representing the relationships between the tables.
            node (str): The connector node that has to be moved.
            node_original_community_id (int): The ID of the community in which the node was originally in.
            community_to_move_to (int): The ID of the community to which the node has to be moved.
            max_community_id (int): The maximum ID value which is in the original partition.

        Returns:
            dict: A modified partition in which the node has been moved to a different community, if required. 
                The keys represent the nodes and the values represent the community ID.
        """

        partition = self.partition

        # Gets all nodes in the original community from the partition
        nodes_in_community = [node for node, community_id in partition.items() if community_id == node_original_community_id]
        
        # Remove the connector node from its original community if it has to be moved to a different community
        if community_to_move_to != node_original_community_id and node in nodes_in_community:
            nodes_in_community = [x for x in nodes_in_community if x != node]

        partition[node] = community_to_move_to
        connected_components = graph.get_connected_components(nodes_in_community)

        if connected_components:
            for group in connected_components:
                for node in group:
                    partition[node] = max_community_id + 1
                max_community_id = max_community_id + 1
        
        return partition

    def move_connector_nodes(self, graph, embeddings_dict, check_neighboring_nodes_only=False):
        """
        Identify the connector nodes and moves them 

        Args:
            graph (networkx.Graph): The graph representing the relationships between the tables.
            embeddings_dict (dict): A dictionary with table names as they key and its embeddings as the value.
            check_neighboring_nodes_only (Boolean): A value indicating whether the algorithm should just check for neighboring nodes when comparing 
                or neighboring communities instead. Defaults to False, indicating the check should be of the entire neighboring communities. 
        
        Returns:
            dict: A dictionary in which the keys represent the table names and the values represent the ID of community in which the tables belong.
        """
        
        partition = self.partition

        for node in self.neighbor_count_by_connector_nodes:
            highest_similarity_score = 0
            
            node_original_community_id = partition[node]
            max_value_key = max(partition, key=partition.get)
            max_community_id = partition[max_value_key]

            if check_neighboring_nodes_only:
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