"""
Flask backend server for handling database retrieval and query submission.

Endpoints:
- GET /databases: Retrieve the list of databases from the MySQL server.
- POST /submit: Handle the submission of the selected database and SQL query, including file upload.
"""

import csv
import igraph as ig
import mysql.connector
import os
import random

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from os import getenv, makedirs
from sentence_transformers import SentenceTransformer

from community_graph import Graph
from sub_graph_analyzer import SubGraphAnalyzer
from label_dkd_matcher import Matcher
from table_community import Community
from utils.db import get_nodes_and_edges_from_db
from utils.embeddings import get_embeddings_dict
from utils.general import read_lines, reset_community_id_numbers


# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from the React frontend

app.config['ALLOWED_EXTENSIONS'] = {'csv', 'json'}

# MySQL connection configuration from environment variables
db_config = {
    'host': os.getenv('MYSQL_HOST'),
    'user': os.getenv('MYSQL_USER'),
    'password': os.getenv('MYSQL_PASSWORD'),
    'port': os.getenv('MYSQL_PORT')
}

# Ensure upload directory exists
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/databases', methods=['GET'])
def get_databases():
    """
    Retrieve the list of databases from the MySQL server.

    Returns:
        JSON response containing the list of databases or an error message.
    """
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        cursor.execute('SHOW DATABASES')
        databases = [db[0] for db in cursor.fetchall()]
        cursor.close()
        connection.close()
        return jsonify(databases), 200
    except mysql.connector.Error as err:
        print(f'Error: {err}')
        return jsonify({'error': str(err)}), 500

@app.route('/submit', methods=['POST'])
def submit_data():
    """
    Handle the submission.

    Returns:
        JSON response confirming receipt of the data.
    """
    
    database = request.form.get('database')
    data_dictionary = request.files.get('data_dictionary')
    document_objects = request.files.get('document_objects')
    community_detection_algorithm = request.form.get('communityAlgorithm')
    similarity_measure = request.form.get('similarityMeasure')
    llm = request.form.get('llm')
    raw_description = request.form.get('rawDescription')
    punctuations = request.form.get('punctuations')
    stopwords = request.form.get('stopwords')
    stemming = request.form.get('stemming')

    response = {}

    # Check and save the data_dictionary file
    if data_dictionary and allowed_file(data_dictionary.filename):
        data_dict_path = os.path.join(UPLOAD_FOLDER, 'data_dictionary.csv')
        data_dictionary.save(data_dict_path)
        response['data_dictionary'] = f'File saved as {data_dict_path}'
    else:
        response['data_dictionary'] = 'No valid CSV file uploaded'
    
    # Check and save the document_objects file
    if document_objects and allowed_file(document_objects.filename):
        doc_obj_path = os.path.join(UPLOAD_FOLDER, 'data_documents.json')
        document_objects.save(doc_obj_path)
        response['document_objects'] = f'File saved as {doc_obj_path}'
    else:
        response['document_objects'] = 'No valid JSON file uploaded'

    conn = mysql.connector.connect(**db_config)

    SECTION = os.getenv('SECTION')
    MODEL = os.getenv('MODEL_NAME')
    TRANSACTION_TABLES_ONLY = False if os.getenv('TRANSACTION_TABLES_ONLY').lower() in ['false', '0'] else True
    USE_OPENAI = False if os.getenv('USE_OPENAI').lower() in ['false', '0'] else True
    OPENAI_MODEL = os.getenv('OPENAI_MODEL_NAME')

    model = OPENAI_MODEL if USE_OPENAI else SentenceTransformer(MODEL)

    embeddings_dict = {}
    descriptions = {}
    
    # Create embeddings of the descriptions
    with open('uploads/data_dictionary.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header line

        for row in reader:
            table_name = row[0]
            description = row[1]
            embeddings_dict = get_embeddings_dict(table_name, description, model, embeddings_dict, USE_OPENAI)
            descriptions[table_name] = description
    nodes, edges = get_nodes_and_edges_from_db(conn, db_type=SECTION, db_name=database)

    TRANSACTION_TABLES_ONLY = False if os.getenv('TRANSACTION_TABLES_ONLY').lower() in ['false', '0'] else True
    if TRANSACTION_TABLES_ONLY:
        TRANSACTION_TABLES_FILENAME = os.getenv('TRANSACTION_TABLES_FILENAME')
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
    
    # COMMUNITY_DETECTION_ALGORITHM = os.getenv('COMMUNITY_DETECTION_ALGORITHM')
    community = Community({})
    # communities = community.get_communities(G_igraph, algorithm=COMMUNITY_DETECTION_ALGORITHM)
    communities = community.get_communities(G_igraph, algorithm=community_detection_algorithm)
 
    original_partition = {node['name']: membership for node, membership in zip(G_igraph.vs, communities.membership)}
    
    # Convert igraph to networkx
    G = Graph()
    G.convert_igraph_to_networkx_graph(G_igraph, original_partition)

    G.display_graph(original_partition)
    
    community = Community(original_partition)
    
    # Move the connector nodes
    modified_partition = community.move_connector_nodes(G, embeddings_dict, similarity_measure, check_neighboring_nodes_only=False)
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
        partition = subgraph_analyzer.move(embeddings_dict, similarity_measure)

        partition, count = reset_community_id_numbers(partition, count) # Reset the count of IDs so that all consecutive numbers are present 
        final_partition.append(partition)

    final_partition_dict = {}
    for i in final_partition:
        final_partition_dict.update(i)

    final_partition_dict, count = reset_community_id_numbers(final_partition_dict)  # Reset the count of IDs so that all consecutive numbers are present
   
    G.display_graph(final_partition_dict, save=False)
 
    COMMUNITY_LABELS_FILENAME = os.getenv('COMMUNITY_LABELS_FILENAME')

    matcher = Matcher()
    matcher.get_documents_from_dkd('uploads/data_documents.json')
    matcher.get_community_labels(COMMUNITY_LABELS_FILENAME)
    similarity_scores = matcher.compute_similarity_scores(model, similarity_measure, USE_OPENAI)
    response['similarity_scores'] = similarity_scores

    return jsonify({'status': 'success', 'data': response}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
