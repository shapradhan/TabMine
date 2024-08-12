import csv
import random

from community import community_louvain as cl
from db_connector import connect
from dotenv import load_dotenv
from os import getenv
from sentence_transformers import SentenceTransformer

import igraph as ig
import networkx as nx

from community_graph import Graph
from sub_graph_analyzer import SubGraphAnalyzer
from label_dkd_matcher import Matcher
from table_community import Community
from utils.db import get_nodes_and_edges_from_db
from utils.embeddings import get_embeddings_dict
from utils.general import read_lines, reset_community_id_numbers

if __name__ == '__main__':
    load_dotenv()

    DB_CONFIG_FILENAME = getenv('DB_CONFIG_FILENAME')
    SECTION = getenv('SECTION')
    RAW_DESCRIPTIONS_FILEPATH = getenv('RAW_DESCRIPTIONS_FILEPATH')
    DB_NAME = getenv('DB_NAME')
    MODEL = getenv('MODEL_NAME')
    TRANSACTION_TABLES_ONLY = False if getenv('TRANSACTION_TABLES_ONLY').lower() in ['false', '0'] else True
    USE_OPENAI = False if getenv('USE_OPENAI').lower() in ['false', '0'] else True
    OPENAI_MODEL = getenv('OPENAI_MODEL_NAME')
    SIMILARITY_MEASURE = getenv('SIMILARITY_MEASURE')
    COMMUNITY_DETECTION_ALGORITHM = getenv('COMMUNITY_DETECTION_ALGORITHM')

    model = OPENAI_MODEL if USE_OPENAI else SentenceTransformer(MODEL)

    preprocessed_texts = {}
    embeddings_dict = {}
    descriptions = {}
    
    # Create embeddings of the descriptions
    with open(RAW_DESCRIPTIONS_FILEPATH, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header line

        for row in reader:
            table_name = row[0]
            description = row[1]
            embeddings_dict = get_embeddings_dict(table_name, description, model, embeddings_dict, USE_OPENAI)
            descriptions[table_name] = description

    # Connect with the database
    conn = connect(filename = DB_CONFIG_FILENAME, section = SECTION)
    nodes, edges = get_nodes_and_edges_from_db(conn, db_type=SECTION, db_name=DB_NAME)
    
    if TRANSACTION_TABLES_ONLY:
        TRANSACTION_TABLES_FILENAME = getenv('TRANSACTION_TABLES_FILENAME')
        nodes_to_keep = read_lines(TRANSACTION_TABLES_FILENAME)
        
        filtered_edges = [edge for edge in edges if all(node in nodes_to_keep for node in edge)]
        nodes = nodes_to_keep
        edges = filtered_edges

    # Set a random seed for reproducibility
    seed = 42
    random.seed(seed)

    G_igraph = ig.Graph()
    G_igraph.add_vertices(nodes)
    G_igraph.add_edges(edges)

    community = Community({})
    communities = community.get_communities(G_igraph, algorithm=COMMUNITY_DETECTION_ALGORITHM)

    original_partition = {node['name']: membership for node, membership in zip(G_igraph.vs, communities.membership)}

    # Convert igraph to networkx
    G = Graph()

    # Add nodes with community membership as an attribute
    for node in G_igraph.vs:
        G.add_node(node['name'], community=original_partition[node['name']])

    # Add edges to the networkx graph
    for edge in G_igraph.es:
        source, target = edge.tuple
        G.add_edge(G_igraph.vs[source]['name'], G_igraph.vs[target]['name'])
    G.display_graph(original_partition)
    
    community = Community(original_partition)
    neighbor_count_of_connector_nodes = community.get_neighbor_count_of_connector_nodes(G, reverse=False)
    
    # Move the connector nodes
    modified_partition = community.move_connector_nodes(G, embeddings_dict, SIMILARITY_MEASURE, check_neighboring_nodes_only=False)
    modified_partition, count = reset_community_id_numbers(modified_partition)  # Reset the count of IDs so that all consecutive numbers are present 
    G.display_graph(modified_partition)
    modified_community = Community(modified_partition)
    modified_nodes_by_community = modified_community.group_nodes_by_community()
    
    nodes = list(G.nodes())
    edges = list(G.edges())

    final_partition = []
    count = 0

    # Traverse through each community and find additional communities 
    for community_id, nodes_to_include in modified_nodes_by_community.items():
        filtered_edges = [edge for edge in edges if edge[0] in nodes_to_include and edge[1] in nodes_to_include]
        subgraph = G.subgraph(nodes_to_include + [u for u, v in filtered_edges] + [v for u, v in filtered_edges])

        subgraph_analyzer = SubGraphAnalyzer(subgraph)
        partition = subgraph_analyzer.move(embeddings_dict, SIMILARITY_MEASURE)

        partition, count = reset_community_id_numbers(partition, count) # Reset the count of IDs so that all consecutive numbers are present 
        final_partition.append(partition)

    final_partition_dict = {}
    for i in final_partition:
        final_partition_dict.update(i)

    final_partition_dict, count = reset_community_id_numbers(final_partition_dict)  # Reset the count of IDs so that all consecutive numbers are present
   
    G.display_graph(final_partition_dict, save=False)
  
    #converted_dict = {str(value): [key for key, val in final_partition_dict.items() if val == value] for value in set(final_partition_dict.values())}
 
    DOCUMENT_LIST_DIR = getenv('DOCUMENT_LIST_DIR')
    DOCUMENT_LIST_FILENAME = getenv('DOCUMENT_LIST_FILENAME')
    COMMUNITY_LABELS_FILENAME = getenv('COMMUNITY_LABELS_FILENAME')

    matcher = Matcher()
    matcher.get_documents_from_dkd(DOCUMENT_LIST_DIR + '/' + DOCUMENT_LIST_FILENAME)
    matcher.get_community_labels(COMMUNITY_LABELS_FILENAME)
    similarity_scores = matcher.compute_similarity_scores(model, SIMILARITY_MEASURE, USE_OPENAI)
    print(similarity_scores)