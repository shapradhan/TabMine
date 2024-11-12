import networkx as nx
from utils.embeddings import calculate_average_similarity

class CommunitySplitter:
    """
    Class to manage and split communities in a network graph based on similarity between connected nodes.

    Attributes:
        partition (dict): Maps node names to community identifiers.
        embeddings_dict (dict): Maps node names to their embeddings.
        graph (nx.Graph): A NetworkX graph of nodes and their edges.
    """
    def __init__(self, partition, embeddings_dict, graph):
        self.partition = partition
        self.embeddings_dict = embeddings_dict
        self.graph = graph

    def find_least_similar_edges(self):
        """
        Identifies and sorts edges within the same community based on similarity.

        Returns:
            list: Sorted list of (node1, node2, similarity) tuples, from least to most similar.
        """
        edges = [
            (node1, node2, calculate_average_similarity([self.embeddings_dict[node1], self.embeddings_dict[node2]]))
            for node1, node2 in self.graph.edges()
            if self.partition[node1] == self.partition[node2]
        ]
        return sorted(edges, key=lambda x: x[2])

    def find_connected_nodes(self, start_node, exclude_edge):
        """
        Finds all nodes connected to a start node within the same community, excluding a specific edge.

        Args:
            start_node (str): Starting node.
            exclude_edge (tuple): Edge to exclude from traversal.

        Returns:
            set: Connected nodes within the same community, excluding the specified edge.
        """
        visited = set()
        to_visit = [start_node]
        start_partition_id = self.partition[start_node]

        while to_visit:
            node = to_visit.pop()
            if node not in visited:
                visited.add(node)
                for neighbor in self.graph.neighbors(node):
                    if {node, neighbor} != set(exclude_edge) and self.partition[neighbor] == start_partition_id:
                        to_visit.append(neighbor)

        return visited
    

    def split_communities(self, community_id):
        """
        Splits communities based on the least similar edges while considering connected nodes
        within the specified community.

        Args:
            community_id (int): The ID of the community to be split.

        Returns:
            dict: Updated partition after splitting communities.
        """
        community_nodes = [node for node, cid in self.partition.items() if cid == community_id]

        if not community_nodes:
            print(f"No nodes found in community {community_id}.")
            return self.partition

        edges = self.find_least_similar_edges()

        for node1, node2, sim in edges:
            community1 = self.find_connected_nodes(node1, (node1, node2))
            community2 = self.find_connected_nodes(node2, (node1, node2))

            if len(community1) == 1 and len(community2) == 1:
                print(f"Skipping edge ({node1}, {node2}) because both sides have only one node.")
                continue

            # Handle single-node case for community1
            if len(community1) == 1:
                node = list(community1)[0]
                without_node = [self.embeddings_dict[n] for n in community2 if n != node]
                avg_sim_without = calculate_average_similarity(without_node)
                avg_sim_with = calculate_average_similarity([self.embeddings_dict[node]] + without_node)

                if avg_sim_without > avg_sim_with:
                    self.partition[node] = max(self.partition.values()) + 1  # Move node to new community

            # Handle single-node case for community2
            if len(community2) == 1:
                node = list(community2)[0]
                without_node = [self.embeddings_dict[n] for n in community1 if n != node]
                avg_sim_without = calculate_average_similarity(without_node)
                avg_sim_with = calculate_average_similarity([self.embeddings_dict[node]] + without_node)

                if avg_sim_without > avg_sim_with:
                    self.partition[node] = max(self.partition.values()) + 1  # Move node to new community

            # Handle multi-node case for both community1 and community2
            if len(community1) > 1 and len(community2) > 1:
                left_embeddings = [self.embeddings_dict[n] for n in community1]
                right_embeddings = [self.embeddings_dict[n] for n in community2]
                combined_embeddings = left_embeddings + right_embeddings

                avg_left = calculate_average_similarity(left_embeddings)
                avg_right = calculate_average_similarity(right_embeddings)
                avg_combined = calculate_average_similarity(combined_embeddings)

                if avg_left > avg_combined:
                    new_id = max(self.partition.values()) + 1
                    for n in community1:
                        self.partition[n] = new_id
                elif avg_right > avg_combined:
                    new_id = max(self.partition.values()) + 1
                    for n in community2:
                        self.partition[n] = new_id

        return self.partition

    def find_least_similar_edge_in_cycle(self, cycle_edges):
        """
        Finds the edge with the lowest similarity in a given cycle.

        Args:
            cycle_edges (list): Edges in the cycle.

        Returns:
            tuple: Edge (node1, node2) with the lowest similarity.
        """
        return min(
            cycle_edges,
            key=lambda edge: calculate_average_similarity([self.embeddings_dict[edge[0]], self.embeddings_dict[edge[1]]])
        )
    
    def remove_least_similar_edges_until_acyclic(self):
        """
        For each community, removes the least similar edges until the community graph is acyclic.

        Returns:
            dict: Updated partition dictionary after removing cycles and splitting communities.
        """

        print("----- FINDING ADDITIONAL COMMUNITIES -----")
        partition_subgraphs = {}  # Dictionary to hold subgraphs of each community
        # Group nodes by their community ID
        for node, community_id in self.partition.items():
            partition_subgraphs.setdefault(community_id, []).append(node)

        # Iterate through each community's nodes and process their subgraph
        for community_id, nodes in partition_subgraphs.items():
            # Create a subgraph for the current community
            subgraph = self.graph.subgraph(nodes).copy()

            # If the subgraph is already a tree (no cycles), split the community
            if nx.is_tree(subgraph):
                self.split_communities(community_id)
            else:
                # Remove self-loops (edges from a node to itself) if they exist
                for node in list(subgraph.nodes):
                    if subgraph.has_edge(node, node):
                        subgraph.remove_edge(node, node)

                # While the subgraph contains cycles, remove the least similar edges
                while not nx.is_tree(subgraph):
                    try:
                        # Find a cycle in the graph
                        cycle_edges = nx.find_cycle(subgraph)
                        # Identify the least similar edge in the cycle
                        least_similar_edge = self.find_least_similar_edge_in_cycle(cycle_edges)
                        # Remove the least similar edge from the subgraph
                        subgraph.remove_edge(*least_similar_edge)
                    except nx.NetworkXNoCycle:
                        # No more cycles exist in the subgraph, break out of the loop
                        break

                # After ensuring the subgraph is acyclic, split the community
                self.split_communities(community_id)

        # Return the updated partition dictionary after processing all communities
        return self.partition