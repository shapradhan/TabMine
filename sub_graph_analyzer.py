from collections import deque

from utils.embeddings import calculate_average_similarity
from utils.general import  filter_values_by_dictionary

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

    def _filter_neighbor_nodes_for_neighbor_communities(self, node, arranged_nodes):
        """
        Get neighboring communities of nodes that are connected.

        Args:
            node (str): The node for which neighbor communities have to be found.
            arranged_nodes (dict): A dictionary where the keys represent the nodes and the values represented the community ID 
                that has been assigned to a key.

        Returns:
            list[list]: A list of list. The inner list includes nodes that are members of one community.
 
        Example:
            >>> node = 'vttp'
            >>> arranged_nodes = {'likp': 0, 'vblk': 0}
            >>> result = self._filter_neighbor_nodes_for_neighbor_communities(node, arranged_nodes)
            >>> print(result)
            [['likp', 'vblk']]

        Note:
            - This method is intended for internal use within the class and may not be directly accessible from outside the class.
        """

        neighboring_communities = []
        community_ids_to_consider = []

        for n, community_id in arranged_nodes.items():
            if self.graph.has_edge(node, n):
                community_ids_to_consider.append(community_id)
                related_nodes = [n2 for n2, cid in arranged_nodes.items() if cid == community_id and n2 != node]
                neighboring_communities.append(related_nodes)
        return neighboring_communities

    def _modify_embeddings(self, node, embeddings, embeddings_dict, arranged_nodes={}):
        """
        Modify list of embeddings.

        Args:
            node (str): The node whose embeddings should be added to the list of embeddings.
            embeddings_dict (dict): A dictionary where the keys represent the nodes and the values represented embeddings of the descriptions.
            arranged_nodes (dict, optional): A dictionary where the keys represent the nodes and the values represented the community ID 
                that has been assigned to a key. Defaults to an empty dictionary.

        Returns:
            list: A modified list of embeddings.
 
        Example:
            >>> node = 'vbap'
            >>> arranged_nodes {'vbep': 0, 'vbeh': 0}
            >>> embeddings = self._modify_embeddings(node, embeddings, embeddings_dict, arranged_nodes)
        """
        
        if node in arranged_nodes:
            community_id_of_current_node = arranged_nodes[node]
            embeddings.extend([embeddings_dict[n] for n, community_id in arranged_nodes.items() if community_id == community_id_of_current_node])
            return embeddings
    
        embeddings.append(embeddings_dict[node])
        return embeddings

    def _update_similarity_score(self, similarity_score, neighbor, partition, count):
        """
        Update the similarity score

        Args:
            similarity_score (float): A similarity score 
            neighbor (str): The neighbor with which the similarity score is the highest
            partition (dict): A dictionary where keys are nodes and values are the corresponding community IDs.
            count (int): A community ID that can be used if neighbor is not present in the partition.

        Returns:
            tuple: A tuple containing multiple values.
            - The first value (float): The highest similarity score.
            - The second value (str): The neighbor with which the similarity score is the highest.
            - The third value (int): The community ID of the neighbor in the partition if it exists in partition or the value of count.
        """
        
        return similarity_score, neighbor, partition.get(neighbor, count)


    def move(self, embeddings_dict):
        """
        Move the nodes in a subgraph to additional communities.

        In each community, arrange the nodes in the descending order of their degree.
        For each node, break its neighbors into separate communities and find the “sub-community” with which the node is the most similar with.
        Move the node to that community.
        Repeat the process for all nodes.

        Args:
            embeddings_dict (dict): A dictionary where the keys represent the nodes and the values represented embeddings of the descriptions.
        
        Returns:
            dict: A partition dictionary where keys are nodes and values are the corresponding community IDs.
        """

        partition = self.partition

        # Get a list of nodes and its degrees and arrange them in the descending order of their degree
        nodes_with_degrees = [(node, degree) for node, degree in self.graph.degree()]
        sorted_nodes_with_degrees = sorted(nodes_with_degrees, key=lambda x: x[1], reverse=True)
        
        # Initialize dictionaries to store arranged nodes and their community IDs
        arranged_nodes = {}
        arranged_nodes_by_community_id = {}
        count = 0
       
        for node_with_degree in sorted_nodes_with_degrees:
            current_node = node_with_degree[0]
            neighbors_of_current_node = list(self.graph.neighbors(current_node))

            highest_similarity_score = 0
            most_similar_neighbor = None
            most_similar_community = None

            # Traverse through the neighbors of the current node            
            for neighbor in neighbors_of_current_node:
                # Conduct a breadth-first search to find neighboring communities but do not traverse through the current node
                neighboring_communities = self._bfs_with_exclusion(neighbor, current_node)

                # Get a list of nodes from the neighboring communities that have already been arranged
                already_grouped_nodes = filter_values_by_dictionary(neighboring_communities, arranged_nodes)
       
                # If there are nodes that have already been arranged and the neighbor is one of those nodes, 
                # find neighboring communities of the current node that takes into consideration the fact that the neighbor has already been arranged
                # and hence might have its own set of neighboring nodes in the community it has been grouped into.
                if already_grouped_nodes and neighbor in already_grouped_nodes:
                    neighboring_communities = self._filter_neighbor_nodes_for_neighbor_communities(current_node, arranged_nodes)

                #If neighbor has not been grouped already, perform a breadth-first search but do not traverse through the current node or any already grouped nodes
                elif neighbor not in already_grouped_nodes:
                    neighboring_communities = self._bfs_with_exclusion(neighbor, current_node, already_grouped_nodes)
                    neighboring_communities = [neighboring_communities]

                for community in neighboring_communities:
                    embeddings =  [embeddings_dict[n] for n in community]
                    
                    # Add the current node being checked if the neighboring community has only one member
                    # and calculate the similarity score between the current node and that other node.
                    if len(embeddings) == 1:
                        embeddings = self._modify_embeddings(current_node, embeddings, embeddings_dict, arranged_nodes)
                        similarity_score_with_current_node = calculate_average_similarity(embeddings)

                        if similarity_score_with_current_node > highest_similarity_score:
                            highest_similarity_score, most_similar_neighbor, most_similar_community = self._update_similarity_score(similarity_score_with_current_node, neighbor, partition, count) 

                    # If the number of nodes in the neighboring community is more than 1, then calcualte the similarity score
                    # where the current node is not present in the neighboring community and when the current node is present in the neighboring community.
                    # If the similarity score is higher with the current node being present in the community, then identify the highest similarity score,
                    # the most similar node, and the most similar community.
                    elif len(embeddings) > 1:
                        similarity_score_without_current_node = calculate_average_similarity(embeddings)
                        embeddings = self._modify_embeddings(current_node, embeddings, embeddings_dict, arranged_nodes)
                        similarity_score_with_current_node = calculate_average_similarity(embeddings)
                        difference = similarity_score_with_current_node - similarity_score_without_current_node

                        if difference > 0 and similarity_score_with_current_node > highest_similarity_score:
                            highest_similarity_score, most_similar_neighbor, most_similar_community = self._update_similarity_score(similarity_score_with_current_node, neighbor, partition, count) 

            # If the most similar neighbor has been identified, move the current node and the most similar neighbor to the most similar community. 
            # Change the community IDs in the partition as well.        
            if most_similar_neighbor:
                arranged_nodes[current_node] = most_similar_community
                arranged_nodes[most_similar_neighbor] = most_similar_community
                if most_similar_community in arranged_nodes_by_community_id:
                    arranged_nodes_by_community_id[most_similar_community].append(current_node)
                    arranged_nodes_by_community_id[most_similar_community].append(most_similar_neighbor)
                else:
                    arranged_nodes_by_community_id[most_similar_community] = [current_node, most_similar_neighbor]
                
                partition[current_node] = most_similar_community
                partition[most_similar_neighbor] = most_similar_community

            # Otherwise, use a new id (count) and assign that as the ID to the current node.
            else:
                count = count + 1
                arranged_nodes[current_node] = count
                arranged_nodes_by_community_id[count] = current_node
                partition[current_node] = count

            count = count + 1    
        return self.partition
    