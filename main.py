from dotenv import load_dotenv
from os import getenv

from data_extractor import get_all_fks, get_table_names
from db_connector import connect
from utils import get_edges, draw_graph

def initialize_graph(nodes, edges, directed=False):
  if directed:
    G = nx.DiGraph()
  else:
    G = nx.Graph()
  G.add_nodes_from(nodes)
  G.add_edges_from(edges)
  return G

if __name__ == '__main__':
    load_dotenv()
    
    DB_CONFIG_FILE = getenv('DB_CONFIG_FILE')
    SECTION = getenv('SECTION') 
    
    cur = connect(filename=DB_CONFIG_FILE, section=SECTION)
    all_fk_rels = get_all_fks(cur)
    
    nodes = [i[0] for i in get_table_names(cur)]
    edges = [get_edges(i) for i in all_fk_rels]
    und_G = initialize_graph(nodes, edges, directed=False)
