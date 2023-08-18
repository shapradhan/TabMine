import pandas as pd
import random
import tensorflow_hub as hub

from community import community_louvain as cl
from dotenv import load_dotenv
from os import getenv, path

from data_extractor import get_all_fks, get_table_names
from db_connector import connect

from community_labeler import labeler
from connecting_node_mover import move_connecting_nodes
from serial_nodes_handler import detect_communities_in_node_series
from utils.embeddings import get_embeddings_dict
from utils.general import create_dict_from_df
from utils.graph import arrange_nodes_in_series, convert_communities_list_to_partition, draw_graph, find_central_node, \
  find_communities_connecting_nodes, get_edges, get_relevant_edges, group_nodes_by_community, initialize_graph, \
  check_any_node_more_than_two_outgoing_edges, find_additional_communities

class ArgumentNotFoundError(Exception):
    pass

if __name__ == '__main__':
  load_dotenv()
    
  DB_CONFIG_FILENAME = getenv('DB_CONFIG_FILENAME')
  SECTION = getenv('SECTION') 
  DESCRIPTIONS_DIR = getenv('DESCRIPTIONS_DIR')
  DESCRIPTIONS_FILENAME = getenv('DESCRIPTIONS_FILENAME')
  MODEL_URL = getenv('MODEL_URL')
  SIMILARITY_THRESHOLD = 0.5
  EMBEDDINGS_FOLDER_NAME = 'embeddings'
  
  cur = connect(filename = DB_CONFIG_FILENAME, section = SECTION)
  foreign_key_relation_list = get_all_fks(cur)
  
  nodes = [i[0] for i in get_table_names(cur)]
  edges = [get_edges(i) for i in foreign_key_relation_list]

  # Set a random seed for reproducibility
  seed = 42
  random.seed(seed)

  G = initialize_graph(nodes, edges, directed=False)
  
  original_partition = cl.best_partition(G)
  # partition = cl.best_partition(G, resolution=1.0)

  # Example partition
  # partition = {
  #   'vbrk': 0, 'bkpf': 0, 'likp': 1, 'lips': 1, 'bseg': 0, 'vbak': 3, 
  #   'vbap': 1, 'cdhdr': 2, 'nast': 3, 'cdpos': 2, 'vbfa': 1, 'vbrp': 0
  #   }

  print('Original partition: {0}'.format(original_partition))
  draw_graph(G, original_partition, title='Original Communities - Undirected - Louvain', color_map='Pastel1')

  # Arrange nodes by community and use that arrangement to get edges and their connecting nodes
  nodes_by_community = group_nodes_by_community(original_partition)
  communities_connecting_nodes = find_communities_connecting_nodes(G, nodes_by_community)

  # Read descriptions file and create a dataframe 
  descriptions_file_path = path.join(DESCRIPTIONS_DIR, DESCRIPTIONS_FILENAME)
  df = pd.read_csv(descriptions_file_path)

  # Create dictionary of descriptions
  descriptions_dict = create_dict_from_df(df, key_col='tables', value_col='descriptions')
  
  # Load the embeddings model
  model = hub.load(MODEL_URL)

  # Get dictionary of embeddings
  embeddings_dict = get_embeddings_dict(descriptions_dict, model, EMBEDDINGS_FOLDER_NAME)

  # communities = {'vbrp': 0, 'vbrk': 0, 'bkpf': 1, 'bseg': 1, 'likp': 2, 'lips': 2, 'vbfa': 3, 'vbap': 3, 'vbak': 3, 'nast': 4, 'cdhdr': 5, 'cdpos': 5}
  #parition = {'ekkn':0 ,'ekes':0,'eket':0,'ekbe':0,'ekko':0,'ekpo':0, 'eina':4, 'eine':4, 'mkpf':3, 'mseg':3, 'eord':5, 'bkpf':1, 'bseg':1, 'rbkp':1, 'rseg':1, 'eban':2, 'ebkn':2}
  
  modified_partition = move_connecting_nodes(original_partition, nodes_by_community, communities_connecting_nodes, embeddings_dict)
  # Example modified_partition = {'vbrk': 0, 'bkpf': 0, 'likp': 1, 'lips': 1, 'bseg': 0, 'vbak': 1, 'vbap': 1, 'cdhdr': 2, 'nast': 3, 'cdpos': 2, 'vbfa': 1, 'vbrp': 0}
  print('Modified partition after applying first arrangement: {0}'.format(modified_partition))

  # Regroup the nodes by community
  nodes_by_community = group_nodes_by_community(modified_partition)

  # Divide the edges in lists of source and target nodes
  sources = [node_group[0] for node_group in edges]
  targets = [node_group[1] for node_group in edges]

  node_chains = []

  # Arrange the nodes in a proper series
  nodes_have_more_than_two_outgoing_edges = check_any_node_more_than_two_outgoing_edges(edges, nodes_by_community)
  
  if nodes_have_more_than_two_outgoing_edges:
    # If any node has more than two outgoing edges
    temp_community_list = []
    for community_id, nodes in nodes_by_community.items():
      similar_nodes = find_additional_communities(nodes, edges, SIMILARITY_THRESHOLD, embeddings_dict)
      temp_community_list.append(similar_nodes)
    
    final_communities = [community for c in temp_community_list for community in c]
  else:
    # If no node has more than two outgoing edges

    for community_id, nodes in nodes_by_community.items():
      print('arranging node {0} in a series'.format(nodes))
      if len(nodes) > 1:
        # Get the central node and relevant edges from nodes in a community
        central_node = find_central_node(nodes, sources, targets, G, centrality_measure='betweenness')
        relevant_edges = get_relevant_edges(nodes, edges)

        # If there are more than one relevant edge, apply arrange_nodes_in_series to get a proper order of the nodes.
        # Otherwise, append the nodes to node_chain. 
        # Only one relevant chain means that there is only one edge that is represented by one tuple containing a source and a target node.
        if len(relevant_edges) != 1:
          nodes_in_series = arrange_nodes_in_series(relevant_edges)
          node_chains.append(nodes_in_series)
        else:
          node_chains.append(nodes)
      else:
        node_chains.append(nodes)
    
    # Loop through each chain in node_chains
    # Example node_chains = [
    #   ['vbrp', 'vbrk', 'bkpf', 'bseg'], 
    #   ['likp', 'lips', 'vbfa', 'vbap', 'vbak'], 
    #   ['cdhdr', 'cdpos'], 
    #   ['nast']
    # ]
    nodes_in_new_communities = []
    for chain in node_chains:
      print('detecting community for {0}'.format(chain))
      new_communities = detect_communities_in_node_series(chain, embeddings_dict, SIMILARITY_THRESHOLD)
      nodes_in_new_communities.append(new_communities)
    
    # Example nodes_in_new_communities = [
    #   [['vbrp', 'vbrk'], ['bkpf', 'bseg']], 
    #   [['likp', 'lips'], ['vbfa'], ['vbap', 'vbak']], 
    #   [['cdhdr', 'cdpos']], 
    #   ['nast']
    # ]

    final_communities = [node for community in nodes_in_new_communities for node in community]
  
  final_partition = convert_communities_list_to_partition(final_communities)
  print('FINAL PARTITION', final_partition )

  new_G = initialize_graph(nodes, edges, directed=False)

  community_labels = labeler(final_partition, descriptions_dict, embeddings_dict)
  # Example community_labels = {0: 'bill', 1: 'account', 2: 'sd delivery', 3: 'sale', 4: 'message status', 5: 'change'}

  draw_graph(new_G, final_partition, title='Modified Communities - Undirected - Louvain', labels= community_labels, color_map='Pastel1')
