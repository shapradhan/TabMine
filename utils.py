import matplotlib.pyplot as plt
import networkx as nx
import os
import re


def _get_parent_node(fk_rel):
    start_str = 'REFERENCES'
    end_str = '('
    result = _get_word_between_strings(fk_rel[2], start_str, end_str).strip()
    return result

def get_edges(fk_rel):
    child_node = fk_rel[0]
    parent_node = _get_parent_node(fk_rel)
    return (parent_node, child_node)

def _get_word_between_strings(text, start_str, end_str):
    pattern = re.compile(rf"{re.escape(start_str)}(.*?){re.escape(end_str)}")
    match = pattern.search(text)
    if match:
        return match.group(1)
    return None

def draw_graph(G, partition, title):
    pos = nx.spring_layout(G)

    plt.figure(figsize=(10, 6))

    for node in G.nodes():
        community_id = partition[node]
        node_color = plt.cm.tab20(community_id)
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=node_color, node_size=200, label=f"Community {community_id}")

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

    plt.title(title)
    plt.legend()
    plt.axis('off')
    plt.show()

def group_nodes_by_community(partition):
    community_nodes = {}
    for node, community_id in partition.items():
        # Get the community_id and make that a key of an empty list, in which the nodes are appended
        community_nodes.setdefault(community_id, []).append(node)
    return community_nodes

def find_connecting_nodes(graph, community_nodes):
    connecting_nodes = {}
    for c1 in community_nodes:
        for c2 in community_nodes:
            if c1 != c2:
                connecting_nodes[(c1, c2)] = set()
                for node in community_nodes[c1]:
                    neighbors = graph.neighbors(node)
                    intersecting_node = set(neighbors).intersection(community_nodes[c2])
                    if intersecting_node:
                        connecting_nodes[(c1, c2)].update(intersecting_node)
    return connecting_nodes

def get_nodes_in_community(partition, community_id):
    return [node for node, comm_id in partition.items() if comm_id == community_id]

def _make_subfolder(subfolder_name):
    if not os.path.exists(subfolder_name):
        os.makedirs(subfolder_name)

def get_string_embedding(input_string, embedding_model):
    embeddings = embedding_model([input_string])
    return embeddings[0]
