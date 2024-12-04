import networkx as nx

from typing import Dict, List, Tuple

from utils.embeddings import calculate_average_similarity

class CommunityManager:
    """
    A class to manage community structure within a graph, including node partitioning and neighbor counts for connector nodes.

    Attributes:
        graph (Graph): The graph containing nodes and edges.
        partition (Dict[int, int]): A dictionary mapping nodes to their respective community IDs.
        neighbor_count_by_connector_nodes (Dict[int, int]): A dictionary mapping connector nodes to the count of their neighbors.
    """

    def __init__(self):
        """
        Initializes the CommunityManager instance with empty attributes.
        """
        self.graph = None
        self.partition = {}
        self.neighbor_count_by_connector_nodes = {}

    def set_graph(self, graph):
        """
        Sets the graph attribute.

        Args:
            graph (Graph): The graph containing nodes and edges.
        """
        self.graph = graph

    def set_partition(self, partition):
        """
        Sets the partition dictionary, mapping nodes to community IDs.

        Args:
            partition (Dict[int, int]): A dictionary mapping nodes to their community IDs.
        """
        self.partition = partition
    

    def get_communities(self, G_igraph, algorithm='multilevel'):
        """
        Detects communities in a graph using the specified community detection algorithm.

        Args:
            G (igraph.Graph): An igraph graph object on which community detection will be performed.
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
            'edge_betweenness': lambda: G_igraph.community_edge_betweenness().as_clustering(3),  # Example for fixed num_communities
            'fastgreedy': lambda: G_igraph.community_fastgreedy().as_clustering(),
            'infomap': lambda: G_igraph.community_infomap(),
            'label_propagation': lambda: G_igraph.community_label_propagation(),
            'leading_eigenvector': lambda: G_igraph.community_leading_eigenvector(),
            'multilevel': lambda: G_igraph.community_multilevel(),
            'spinglass': lambda: G_igraph.community_spinglass(),
            'walktrap': lambda: G_igraph.community_walktrap().as_clustering()
        }
        
        # Default to 'multilevel' if the provided algorithm is not in the map
        if algorithm not in algorithm_map:
            algorithm = 'multilevel'
        
        return algorithm_map[algorithm]()
    
    def get_neighbor_count_of_connector_nodes(self, reverse=True):
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
        neighbor_count = {}
        
        for node1, node2 in self.graph.edges():
            if node1 not in neighbor_count or node2 not in neighbor_count:
                if self.partition[node1] != self.partition[node2]:  # Nodes in different communities
                    neighbor_count[node1] = len(list(self.graph.neighbors(node1)))
                    neighbor_count[node2] = len(list(self.graph.neighbors(node2)))

        self.neighbor_count_by_connector_nodes = dict(sorted(neighbor_count.items(), key=lambda item: item[1], reverse = reverse))

    def _place_node_in_separate_community(self, node: int):
        """
        Assigns a node to a new community if it doesn't fit in its current community.
        
        This method finds the highest existing community ID and assigns the node a new ID in a separate community.
        
        Args:
            node (int): The node to be reassigned to a new community.
        """
        new_community_id = max(self.partition.values(), default=-1) + 1
        self.partition[node] = new_community_id

    def _calculate_similarity_for_original_community(self, node: int, community_id: int, embeddings_dict: Dict[int, List[float]], similarity_measure) -> Tuple[float, float]:
        """
        Calculates similarity scores for the community with and without a connector node.
        
        This method computes the average similarity between embeddings of nodes in the same community both with and without the specified node.
        
        Args:
            node (int): The connector node whose impact on similarity will be calculated.
            community_id (int): The community ID to which the node belongs.
            embeddings_dict (Dict[int, List[float]]): A dictionary mapping nodes to their embeddings.
            similarity_measure (str): Specifies the similarity measure used to evaluate the relationship between nodes. 
                Options include:
                - 'cosine_similarity': Measures the cosine of the angle between two vectors, useful for comparing direction rather than magnitude.
                - 'dot_product': Computes the dot product of two vectors, which gives a measure of similarity based on magnitude and direction.
                - 'euclidean_distance': Calculates the straight-line distance between two points, useful for measuring absolute difference.
                - 'manhattan_distance': Computes the sum of the absolute differences along each dimension, emphasizing grid-like or taxicab distances.

        Returns:
            Tuple[float, float]: The similarity scores for the community with and without the connector node.
        """
        nodes = [n for n in self.partition if self.partition[n] == community_id]
        embeddings = [embeddings_dict[n] for n in nodes]

        sim_with_connector = calculate_average_similarity(embeddings + [embeddings_dict[node]], similarity_measure)
        sim_without_connector = calculate_average_similarity(embeddings, similarity_measure)

        return sim_with_connector, sim_without_connector
    
    def _get_filtered_neighboring_communities(self, node: int, original_community_id: int) -> List[int]:
        """
        Retrieves the unique community IDs of neighboring nodes that are not in the original community.
        
        This method ensures that only the communities of neighboring nodes that are different from the original community are returned.
        
        Args:
            node (int): The node whose neighboring communities are being evaluated.
            original_community_id (int): The community ID of the given node.
        
        Returns:
            List[int]: A list of community IDs for neighboring nodes not in the original community.
        """
        return list({self.partition[neighbor] for neighbor in self.graph.neighbors(node) if self.partition[neighbor] != original_community_id})    

    def _try_move_node_to_best_neighboring_community(self, node: int, original_community_id: int, 
                                                     neighboring_communities: List[int],
                                                     embeddings_dict: Dict[int, List[float]],
                                                     sim_with: float, sim_without: float, similarity_measure,
                                                     max_community_id: int, nodes_to_process: List[int]) -> Tuple[bool, int]:
        """Attempts to move a connector node to the best neighboring community to maximize similarity gain.
        If no valid neighboring community is found, place the node in a new separate community.

        Args:
            node (int): Node to be moved.
            original_community_id (int): Current community ID of the node.
            neighboring_communities (List[int]): Communities neighboring the node.
            embeddings_dict (Dict[int, List[float]]): Embeddings for each node.
            sim_with (float): Similarity score of node's original community with the node.
            sim_without (float): Similarity score of node's original community without the node.
            similarity_measure (str): Specifies the similarity measure used to evaluate the relationship between nodes. 
                Options include:
                - 'cosine_similarity': Measures the cosine of the angle between two vectors, useful for comparing direction rather than magnitude.
                - 'dot_product': Computes the dot product of two vectors, which gives a measure of similarity based on magnitude and direction.
                - 'euclidean_distance': Calculates the straight-line distance between two points, useful for measuring absolute difference.
                - 'manhattan_distance': Computes the sum of the absolute differences along each dimension, emphasizing grid-like or taxicab distances.
            max_community_id (int): Maximum community ID currently used.
            nodes_to_process (List[int]): Queue of nodes to process.

        Returns:
            Tuple[bool, int]: Whether the node was moved, and the updated max community ID.
        """
        best_community_id = None  # Initialize to track the best community to move to
        best_improvement = float('-inf')  # Track the highest similarity improvement found

        # Iterate through each neighboring community to find the best community for the node
        for community_id in neighboring_communities:
            nodes_in_community = self._get_nodes_by_community_id(community_id)  # Get nodes in this community

            # Calculate similarity with the node added to this community
            embeddings = [embeddings_dict[n] for n in nodes_in_community + [node]]
            sim_with_connector = calculate_average_similarity(embeddings)

            # Calculate similarity without the node (only if there’s more than one node in the community)
            embeddings = [embeddings_dict[n] for n in nodes_in_community] if len(nodes_in_community) > 1 else embeddings
            sim_without_connector = calculate_average_similarity(embeddings)

            # Determine if the improvement is better than any previously found
            improvement = sim_with_connector - sim_without_connector
            if sim_without > sim_with and improvement > best_improvement:
                best_improvement = improvement
                best_community_id = community_id  # Update to the current best community

        # Move the node to the best neighboring community if a positive improvement was found
        if best_community_id is not None and best_improvement > 0:
            self.partition[node] = best_community_id  # Update partition to reflect the new community
            # Check for disconnected nodes in the original community and split if necessary
            new_communities, max_community_id = self._check_and_split_disconnected_nodes(original_community_id, max_community_id)

            # Add any newly created communities to the nodes_to_process queue
            for new_community_id in new_communities:
                nodes_to_process = self._update_nodes_to_process(nodes_to_process, new_community_id)

            return True, max_community_id  # Node was moved; return updated max_community_id

        # If no valid neighboring community found, place the node in a new separate community
        self._place_node_in_separate_community(node)
        # Update nodes_to_process with new connector nodes from this community
        nodes_to_process = self._update_nodes_to_process(nodes_to_process, max_community_id)
        
        return False, max_community_id  # Node was not moved; return updated max_community_id

    def _update_nodes_to_process(self, nodes_to_process: List[int], new_community_id: int) -> List[int]:
        """Updates the processing queue with new connector nodes formed in a community.

        Args:
            nodes_to_process (List[int]): List of nodes to process.
            new_community_id (int): ID of the newly formed community.

        Returns:
            List[int]: Updated list of nodes to process.
        """
        # Use a set for quick lookups and to avoid duplicate entries in nodes_to_process
        nodes_to_process_set = set(nodes_to_process)
        
        # Identify nodes in the new community
        new_community_nodes = [n for n, comm_id in self.partition.items() if comm_id == new_community_id]
        
        # Check each node in the new community for potential as a connector node
        for new_node in new_community_nodes:
            for neighbor in self.graph.neighbors(new_node):
                # If the neighbor belongs to a different community, new_node is a connector
                if self.partition[neighbor] != new_community_id:
                    nodes_to_process_set.add(new_node)
                    break  # No need to check more neighbors for this node

        return list(nodes_to_process_set)
    
    def _check_and_split_disconnected_nodes(self, original_community_id: int, max_community_id: int) -> Tuple[List[int], int]:
        """Identifies disconnected components in the original community and assigns new community IDs to each component.

        Args:
            original_community_id (int): The ID of the original community to check.
            max_community_id (int): The current maximum community ID.

        Returns:
            Tuple[List[int], int]: A list of new community IDs formed, and the updated max community ID.
        """
        # Retrieve all nodes that are part of the original community
        nodes_in_original_community = [n for n, comm_id in self.partition.items() if comm_id == original_community_id]
        
        # Create a subgraph with only nodes in the original community to identify connected components
        subgraph = self.graph.subgraph(nodes_in_original_community)
        connected_components = list(nx.connected_components(subgraph))

        new_communities = []  # Track newly created community IDs

        for component in connected_components:
            # Assign a new community ID to each component
            new_community_id = max_community_id + 1
            max_community_id += 1
            for node in component:
                self.partition[node] = new_community_id
            new_communities.append(new_community_id)

        return new_communities, max_community_id
    
    def _get_nodes_by_community_id(self, community_id):
        """
        Retrieve all nodes associated with a specified community ID.

        Args:
            community_id (int): The ID of the community for which nodes are to be retrieved.

        Returns:
            list: A list of nodes that belong to the specified community.
        """
        return [node for node, comm_id in self.partition.items() if comm_id == community_id]

    def move_connector_nodes(self, embeddings_dict: Dict[int, List[float]], similarity_measure) -> Dict[int, int]:
        """
        Reassigns connector nodes between communities to maximize similarity within each community.

        Args:
            embeddings_dict (dict): Dictionary mapping each node to its embedding vector.
            similarity_measure (str): Specifies the similarity measure used to evaluate the relationship between nodes. 
                Options include:
                - 'cosine_similarity': Measures the cosine of the angle between two vectors, useful for comparing direction rather than magnitude.
                - 'dot_product': Computes the dot product of two vectors, which gives a measure of similarity based on magnitude and direction.
                - 'euclidean_distance': Calculates the straight-line distance between two points, useful for measuring absolute difference.
                - 'manhattan_distance': Computes the sum of the absolute differences along each dimension, emphasizing grid-like or taxicab distances.

        Returns:
            dict: Updated partition mapping each node to a community.
        """
        print("----- MOVING CONNECTOR NODES -----")

        # Find the highest community ID currently in use to generate new unique IDs if needed.
        max_community_id = max(self.partition.values(), default=-1)
        
        # Initialize the list of connector nodes to process, starting with nodes from neighbor count data.
        nodes_to_process = list(self.neighbor_count_by_connector_nodes.keys())

        # Loop through each connector node to evaluate and potentially reassign it to a better community.
        while nodes_to_process:
            # Process nodes in a FIFO order by taking the first node.
            node = nodes_to_process.pop(0)
            
            # Retrieve the current community ID of the node.
            original_community_id = self.partition[node]

            # Calculate similarity scores for the node's current community with and without the node.
            sim_with, sim_without = self._calculate_similarity_for_original_community(
                node, original_community_id, embeddings_dict, similarity_measure
            )

            # If the community's similarity score is not improved by keeping the node:
            if sim_with <= sim_without:
                # Identify neighboring communities for potential reassignment.
                neighboring_communities = self._get_filtered_neighboring_communities(node, original_community_id)

                # Attempt to move the node to the best neighboring community based on similarity improvements.
                node_moved, max_community_id = self._try_move_node_to_best_neighboring_community(
                    node, original_community_id, neighboring_communities, embeddings_dict,
                    sim_with, sim_without, similarity_measure, max_community_id, nodes_to_process
                )

                # If no better community was found, assign the node to a new separate community.
                if not node_moved and self.partition[node] == original_community_id:
                    self._place_node_in_separate_community(node)

                    # Update nodes to process with any new connector nodes formed due to reassignment.
                    nodes_to_process = self._update_nodes_to_process(nodes_to_process, max_community_id)

        # Return the updated partition of nodes to communities.
        return self.partition