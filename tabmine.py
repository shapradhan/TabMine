import argparse
import csv
import igraph as ig
import json
import os
import pandas as pd
import random

from dotenv import load_dotenv

from community_graph import Graph
from community_manager import CommunityManager
from community_splitter import CommunitySplitter
from labeler import Labeler
from table_columns_manager import TableColumnsManager
from table_scorer import TableScorer
from utils.db import create_connnection, get_nodes_and_edges_from_db
from utils.embeddings import create_embeddings
from utils.general import include_nodes_and_edges, reset_community_id_numbers


def parse_input_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Command-line app for database configuration and algorithm selection")

    # Add arguments for the required inputs
    parser.add_argument('--dbtype', type=str, required=True, help="Type of database (e.g., MySQL, PostgreSQL)")
    parser.add_argument('--dbname', type=str, required=True, help="Name of the database")
    parser.add_argument('--data_dictionary', type=str, required=True, help="File path of the CSV file")
    parser.add_argument('--business_documents', type=str, required=True, help="File path of the JSON file")
    parser.add_argument('--community_detection', type=str, required=True, help="Community detection algorithm (e.g., Louvain, Girvan-Newman)")
    parser.add_argument('--similarity_measure', type=str, required=True, help="Similarity measure (e.g., Jaccard, Cosine)")

    # Parse the arguments
    args = parser.parse_args()

    return args

def read_data_dictionary_and_get_embeddings(data_dict_path, use_short_descriptions, preprocessing_options):
    embeddings_dict = {}
    descriptions = {}
    
    with open(data_dict_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header line
        
        for row in reader:
            table_name = row[0]
            description = row[1]
            
            if use_short_descriptions:
                labeler = Labeler({}, {})
                short_description = labeler.create_short_description(table_name, description)
                descriptions[table_name] = short_description

                embeddings_dict = create_embeddings(table_name, short_description, embeddings_dict, preprocessing_options)

            else:
                descriptions[table_name] = description
                embeddings_dict = create_embeddings(table_name, description, embeddings_dict, preprocessing_options)

    return descriptions, embeddings_dict

def create_communities(load_final_partition_from_file, use_subset_tables, connection, db_type, database, community_detection_algorithm, embeddings_dict):
    partition_files_dir = os.getenv('PARTITION_FILES_DIR')
    original_partition_filename = os.getenv('ORIGINAL_PARTITION_FILENAME')
    modified_partition_filename = os.getenv('MODIFIED_PARTITION_FILENAME')
    final_partition_filename = os.getenv('FINAL_PARTITION_FILENAME')
    image_save_dir = os.getenv('IMAGE_SAVE_DIR')

    # Make partition_files_dir if it does not exist
    os.makedirs(partition_files_dir, exist_ok=True)

    original_partition_path = os.path.join(partition_files_dir, original_partition_filename)
    modified_partition_path = os.path.join(partition_files_dir, modified_partition_filename)
    final_partition_path = os.path.join(partition_files_dir, final_partition_filename)

    if load_final_partition_from_file:
        print("-----------------Loading partition from file----------------------")
        
        with open(final_partition_path, "r") as file:
            final_partition = json.load(file)
        nodes, edges = get_nodes_and_edges_from_db(connection, db_type, db_name=database)
        G = Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
            
    else:
        print("-----------------Creating communities----------------------")
        
        nodes, edges = get_nodes_and_edges_from_db(connection, db_type, db_name=database)

        if use_subset_tables:
            df = pd.read_csv("all_table_base_and_sales_only.csv")
            tables_to_include = df["tables"].tolist()
            nodes, edges = include_nodes_and_edges(nodes, edges, tables_to_include)

        # Set a random seed for reproducibility
        seed = 42
        random.seed(seed)

        G_igraph = ig.Graph()
        G_igraph.add_vertices(nodes)
        G_igraph.add_edges(edges)

        community_manager = CommunityManager()
        communities = community_manager.get_communities(G_igraph, algorithm=community_detection_algorithm)
        original_partition = {node['name']: membership for node, membership in zip(G_igraph.vs, communities.membership)}
       
        with open(original_partition_path, "w") as file:
            json.dump(original_partition, file)

        G = Graph()
        G.convert_igraph_to_networkx_graph(G_igraph, original_partition)
        # G.display_graph(original_partition, save=True, filename='original.png', save_dir_path=image_save_dir)

        # Update connector node membership
        community_manager.set_graph(G)
        community_manager.set_partition(original_partition)
        community_manager.get_neighbor_count_of_connector_nodes(reverse=True)


        paritition_after_connector_nodes_updated = community_manager.move_connector_nodes(embeddings_dict, 'cosine_similarity')
        # G.display_graph(paritition_after_connector_nodes_updated, save=True, filename='modified.png', save_dir_path=image_save_dir)

        with open(modified_partition_path, "w") as file:
            json.dump(paritition_after_connector_nodes_updated, file)
        
        # Identify additional communities
        community_splitter = CommunitySplitter(paritition_after_connector_nodes_updated, embeddings_dict, G)
        final_partition = community_splitter.remove_least_similar_edges_until_acyclic()
        final_partition, _ = reset_community_id_numbers(final_partition)

        # G.display_graph(final_partition, save=True, filename='final.png', save_dir_path=image_save_dir)

        with open(final_partition_path, "w") as file:
            json.dump(final_partition, file)

    return G, final_partition

def assign_community_labels(load_community_labels_from_file, final_partition, descriptions):
    business_document_filename = os.getenv('BUSINESS_DOCUMENT_FILENAME')

    labeler = Labeler(final_partition, descriptions)
    labeler.group_nodes_by_community_id()

    if not load_community_labels_from_file:
        print("----------------Creating labels----------------------")
        labeler.group_nodes_by_community_id()
        labeler.create_dict()
        labeler.generate_labels()
        labeler.save_labels()
    
    with open(business_document_filename, 'r') as file:
        biz_docs = json.load(file)
        
    document_names = [doc["name"] for doc in biz_docs["documents"]]
    labeler.set_documents(document_names)
    labels = labeler.get_community_labels('labels.csv')

    return labeler, labels

def find_score_with_fields_and_columns(labeler, connection, G, community_similarity_threshold, field_weight, profile_weight, neighbor_weight):
    business_document_filename = os.getenv('BUSINESS_DOCUMENT_FILENAME')

    sim_scores = labeler.compute_similarity_scores('cosine_similarity')
    table_columns_manager = TableColumnsManager(sim_scores)
    candidate = table_columns_manager.find_candidate_communities(community_similarity_threshold)
    tables = table_columns_manager.get_tables_from_csv(candidate, 'labels.csv')
    table_fields = table_columns_manager.get_table_columns(connection, tables)

    with open(business_document_filename, 'r') as file:
        biz_docs = json.load(file)

    table_scorer = TableScorer(biz_docs, table_fields, G, field_weight, profile_weight, neighbor_weight)
    rel_tables, final_results_sorted = table_scorer.identify_relevant_tables()

    return final_results_sorted

def main():
    # Parse the arguments
    args = parse_input_args()

    db_type = args.dbtype
    database = args.dbname
    data_dict_path = args.data_dictionary
    community_detection_algorithm = args.community_detection

    use_short_descriptions = False
    load_final_partition_from_file = False
    use_subset_tables = False
    load_community_labels_from_file = False

    connection = create_connnection(db_type, database)

    preprocessing_options = {
        'raw_description': True,
        'punctuations': False,
        'stopwords': False,
        'lemmatizing': False
    }

    descriptions, embeddings_dict = read_data_dictionary_and_get_embeddings(data_dict_path, use_short_descriptions, preprocessing_options)

    G, final_partition = create_communities(load_final_partition_from_file, use_subset_tables, connection, db_type, database, community_detection_algorithm, embeddings_dict)
    
    # Assign community labels
    labeler, labels = assign_community_labels(load_community_labels_from_file, final_partition, descriptions)
   
    # Create label embeddings
    label_embeddings = {}

    for community_id, label in labels.items():
        label_embeddings = labeler.load_or_create_embeddings(label, "embeddings/odoo/labels", label_embeddings)
        
    community_similarity_threshold = 0.9
    field_weight = 0.571
    profile_weight = 0.286
    neighbor_weight = 0.143

    final_results_sorted = find_score_with_fields_and_columns(labeler, connection, G, community_similarity_threshold, field_weight, profile_weight, neighbor_weight)

    print('\nFinal results', final_results_sorted)

if __name__ == '__main__':
    load_dotenv()
    main()