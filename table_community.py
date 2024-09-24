from utils.embeddings import calculate_average_similarity
from community_graph import Graph

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
        
        Note:
            - This method is intended for internal use within the class and may not be directly accessible from outside the class.
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
        
        Note:
            - This method is intended for internal use within the class and may not be directly accessible from outside the class.
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

        Note:
            - This method is intended for internal use within the class and may not be directly accessible from outside the class.
        """

        return [node for node, comm_id in self.partition.items() if comm_id == community_id]

    def _get_max_community_id(self, partition):
        """Retrieves the maximum community ID from the partition.

        Args:
            partition (dict): A dictionary where the keys are node identifiers 
                            and the values are integers representing community IDs.

        Returns:
            int: The maximum community ID found in the partition.

        Raises:
            ValueError: If the partition contains non-integer values.
            TypeError: If the maximum community ID is not an integer.
        """
        
        # Ensure that partition stores integer values representing community IDs
        if not all(isinstance(v, int) for v in partition.values()):
            raise ValueError("Partition should only contain integer community IDs.")
        
        max_value_key = max(partition, key=partition.get)  # Get the key with the maximum value
        max_community_id = partition[max_value_key]  # The maximum community ID should be an integer
        
        # Check if the result is an integer, raise error if not
        if not isinstance(max_community_id, int):
            raise TypeError("max_community_id is expected to be an integer.")
        
        return max_community_id

    def _get_filtered_neighboring_communities(self, graph, node, node_original_community_id):
        """Retrieves neighboring communities for a given node, excluding its original community.

        This method first fetches the neighboring communities of the specified node 
        and then filters out the community that the node originally belongs to.

        Args:
            graph (networkx.Graph): The graph representing the relationships between the nodes.
            node (str): The identifier of the node for which neighboring communities are to be found.
            node_original_community_id (int): The community ID of the node's original community, which will be excluded from the results.

        Returns:
            list: A list of community IDs representing the neighboring communities, excluding the original community ID.
        """
        
        neighboring_communities = self._get_neighboring_communities(graph, node)
    
        # Filter out the original community ID from neighboring communities
        neighboring_communities = [community_id for community_id in neighboring_communities if community_id != node_original_community_id]
        return neighboring_communities

    def _calculate_similarity_for_original_community(self, node, node_original_community_id, embeddings_dict, similarity_measure):
        """Calculates the similarity scores for a given node within its original community.

        This method computes the similarity score between the connector node and the nodes in its original community,
        both with and without the connector node included in the calculation. If the original community contains
        two or fewer nodes, the similarity score without the connector node will be the same as the score with it.

        Args:
            node (str): The identifier of the connector node.
            node_original_community_id (int): The community ID of the original community where the node resides.
            embeddings_dict (dict): A dictionary mapping node identifiers to their corresponding embeddings.
            similarity_measure (callable): A function to calculate the similarity score between embeddings.

        Returns:
            tuple: A tuple containing two elements:
                - float: The similarity score with the connector node included.
                - float: The similarity score without the connector node included.
        """
        
        nodes_in_original_community = self._get_nodes_by_community_id(node_original_community_id)
        
        # Calculate similarity score with the connector node
        embeddings_original = [embeddings_dict[n] for n in nodes_in_original_community]
        sim_score_with_connector_in_original = calculate_average_similarity(embeddings_original, similarity_measure)

        # Handle the case where the original community has more than 2 nodes
        if len(nodes_in_original_community) > 2:
            nodes_in_original_community.remove(node)
            embeddings_without_connector_original = [embeddings_dict[n] for n in nodes_in_original_community]
            sim_score_without_connector_in_original = calculate_average_similarity(embeddings_without_connector_original, similarity_measure)
        else:
            sim_score_without_connector_in_original = sim_score_with_connector_in_original

        return sim_score_with_connector_in_original, sim_score_without_connector_in_original
    
    def _try_move_node_to_neighboring_community(self, node, node_original_community_id, neighboring_communities, 
                                            partition, embeddings_dict, sim_score_with_connector_in_original, 
                                            sim_score_without_connector_in_original, similarity_measure, max_community_id):
        """
        Attempts to move a connector node from its original community to one of its neighboring communities based on
        similarity score comparisons. If moved, the function ensures that nodes remaining in the original community 
        remain connected and updates the partition accordingly.

        The node is moved if:
        - The similarity score of the original community improves by removing the node.
        - The similarity score of the neighboring community improves by adding the node.

        Args:
            node (any hashable type): The connector node being evaluated for movement.
            node_original_community_id (int): The ID of the original community to which the node belongs.
            neighboring_communities (list of int): A list of community IDs that are adjacent to the node's original community.
            partition (dict): A dictionary where keys are nodes and values are their community IDs, representing the partitioning of nodes.
            embeddings_dict (dict): A dictionary where keys are nodes and values are their embeddings (vector representations).
            sim_score_with_connector_in_original (float): The similarity score of the original community with the connector node included.
            sim_score_without_connector_in_original (float): The similarity score of the original community without the connector node.
            similarity_measure (callable): A function or method to compute similarity between nodes based on their embeddings.
            max_community_id (int): The current highest community ID in the partitioning, used to assign new communities if nodes are disconnected.

        Returns:
            bool: True if the node was moved to a neighboring community, False if it was not moved.
            int: The updated maximum community ID, which might change if disconnected nodes are split into new communities.

        Behavior:
            - If the node is moved to a neighboring community, the partition is updated with the new community assignment.
            - After moving, checks if the remaining nodes in the original community are still connected. If they are disconnected,
            the function assigns each disconnected group to a new community and updates `max_community_id`.
            - If the node is not moved, the function returns without modifying the partition or `max_community_id`.
        """

        for community_id in neighboring_communities:
            nodes_in_neighboring_community = self._get_nodes_by_community_id(community_id)

            # Calculate similarity score for the neighboring community with the connector node
            nodes_with_connector = nodes_in_neighboring_community + [node]
            embeddings_neighboring_with_connector = [embeddings_dict[n] for n in nodes_with_connector]
            sim_score_with_connector_in_neighbor = calculate_average_similarity(embeddings_neighboring_with_connector, similarity_measure)

            # Handle the case where the neighboring community has more than 1 node
            if len(nodes_in_neighboring_community) > 1:
                embeddings_without_connector_neighbor = [embeddings_dict[n] for n in nodes_in_neighboring_community]
                sim_score_without_connector_in_neighbor = calculate_average_similarity(embeddings_without_connector_neighbor, similarity_measure)
            else:
                sim_score_without_connector_in_neighbor = sim_score_with_connector_in_neighbor

            print('sim_score_with_connector_in_neighbor:', sim_score_with_connector_in_neighbor)
            print('sim_score_without_connector_in_neighbor:', sim_score_without_connector_in_neighbor)

            # Check if the node should be moved to this neighboring community
            if (sim_score_without_connector_in_original > sim_score_with_connector_in_original and
                    sim_score_with_connector_in_neighbor > sim_score_without_connector_in_neighbor):
                print(f'Moving node {node} from community {node_original_community_id} to {community_id}.')
                partition[node] = community_id  # Update the node's community ID

                # After moving, check if the remaining nodes in the original community are still connected
                nodes_in_original_community = [n for n in partition if partition[n] == node_original_community_id]
                
                G = Graph()
                max_community_id = G.check_and_split_disconnected_nodes(nodes_in_original_community, partition, max_community_id)

                return True, max_community_id  # Node was moved, return updated max_community_id

        return False, max_community_id  # Node was not moved, return max_community_id unchanged


    def _place_node_in_separate_community(self, node, partition, sim_score_with_connector_in_original, sim_score_without_connector_in_original):
        """Places a node in a separate community if similarity conditions are met.

        This method evaluates whether a node should be placed in a separate community 
        based on its similarity scores with and without the node included in the original community. 
        If the condition is met, a new community ID is assigned to the node.

        Args:
            node (str): The identifier of the node being considered for placement.
            partition (dict): A dictionary where keys are node identifiers and values are community IDs.
            sim_score_with_connector_in_original (float): The similarity score of the original community 
                                                        with the node included.
            sim_score_without_connector_in_original (float): The similarity score of the original community 
                                                            without the node included.

        Returns:
            None: Updates the partition in place by either assigning a new community ID or 
                keeping the node in its original community.
        """
        # Calculate the maximum community ID from the current partition
        max_community_id = max(partition.values()) if partition else 0  # Handle the case when partition is empty

        # Check if the node should be placed in a separate community
        if sim_score_with_connector_in_original < sim_score_without_connector_in_original:
            new_community_id = max_community_id + 1  # Assign a new community ID
            print(f'Putting node {node} in a separate community with ID {new_community_id}.')
            partition[node] = new_community_id  # Assign the new community ID
        else:
            print(f'Node {node} will stay in the original community {partition[node]}.')


    def move_connector_nodes(self, graph, embeddings_dict, similarity_measure):
        """Moves connector nodes between communities based on similarity scores and community structure.

        This method iterates through connector nodes and attempts to move each one to a neighboring community 
        if certain similarity conditions are met. If the node should not stay in its current community or move 
        to a neighboring one, it is placed in a separate community. The partition is updated based on the 
        movements and similarity evaluations.

        Args:
            graph (networkx.Graph): The graph representing the relationships between nodes (e.g., tables).
            embeddings_dict (dict): A dictionary mapping node identifiers to their corresponding embeddings.
            similarity_measure (str): The similarity measure used to calculate similarity scores (e.g., 'cosine', 'euclidean').

        Returns:
            dict: Updated partition dictionary where keys are node identifiers and values are community IDs.

        Note:
            - This method evaluates nodes using the similarity score of their embeddings both within their original 
            community and in neighboring communities. If a node doesn't fit well in its original or neighboring 
            communities, it may be placed in a new community.
        """

        partition = self.partition
        max_community_id = self._get_max_community_id(partition)  # Initialize max_community_id once at the start

        for node in self.neighbor_count_by_connector_nodes:
            # Get the original community and neighboring communities
            node_original_community_id = partition[node]
            neighboring_communities = self._get_filtered_neighboring_communities(graph, node, node_original_community_id)

            # Calculate similarity scores for the original community
            sim_score_with_connector_in_original, sim_score_without_connector_in_original = self._calculate_similarity_for_original_community(
                node, node_original_community_id, embeddings_dict, similarity_measure)

            print('sim_score_with_connector_in_original:', sim_score_with_connector_in_original)
            print('sim_score_without_connector_in_original:', sim_score_without_connector_in_original)

            # Try moving the node to a neighboring community
            node_moved, max_community_id = self._try_move_node_to_neighboring_community(
                node, node_original_community_id, neighboring_communities, partition, embeddings_dict,
                sim_score_with_connector_in_original, sim_score_without_connector_in_original, similarity_measure, max_community_id)

            # If the node was moved, skip the rest
            if node_moved:
                continue

            # If the node should go to a separate community
            self._place_node_in_separate_community(node, partition, sim_score_with_connector_in_original, sim_score_without_connector_in_original)

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

    def get_communities(self, G, algorithm='multilevel'):
        """
        Detects communities in a graph using the specified community detection algorithm.

        Args:
            G (networkx.Graph): A NetworkX graph object on which community detection will be performed.
            algorithm (str, optional, default='multilevel'): The community detection algorithm to use. Options include:
                - 'edge_betweenness': Uses edge betweenness centrality to detect communities. Requires specifying the number of communities.
                - 'fastgreedy': Uses the fast greedy algorithm for community detection.
                - 'infomap': Uses the Infomap algorithm for detecting communities.
                - 'label_propagation': Uses the Label Propagation algorithm for community detection.
                - 'leading_eigenvector': Uses the leading eigenvector algorithm for community detection.
                - 'multilevel': Uses the Multilevel algorithm for community detection.
                - 'spinglass': Uses the Spinglass algorithm for community detection.
                - 'walktrap': Uses the Walktrap algorithm for community detection.

        Returns:
            communities: A community clustering object representing the detected communities in the graph. The exact type and methods available will depend on the algorithm used.
        
        Notes:
            - For the 'edge_betweenness' algorithm, the number of communities is fixed at 3 in this implementation. 
        """
        
        algorithm_map = {
            'edge_betweenness': lambda: G.community_edge_betweenness().as_clustering(3),  # Example for fixed num_communities
            'fastgreedy': lambda: G.community_fastgreedy().as_clustering(),
            'infomap': lambda: G.community_infomap(),
            'label_propagation': lambda: G.community_label_propagation(),
            'leading_eigenvector': lambda: G.community_leading_eigenvector(),
            'multilevel': lambda: G.community_multilevel(),
            'spinglass': lambda: G.community_spinglass(),
            'walktrap': lambda: G.community_walktrap().as_clustering()
        }
        
        # Default to 'multilevel' if the provided algorithm is not in the map
        if algorithm not in algorithm_map:
            algorithm = 'multilevel'
        
        return algorithm_map[algorithm]()