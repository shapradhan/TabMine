from data_extractor import get_all_fks, get_table_names as get_table_names_p
from mysql_data_extractor import get_all_foreign_key_relationships, get_table_names as get_table_names_m
from utils.general import extract_substring_between_strings

def _get_parent_table(foreign_key_relation_list):
    """ 
    Extract the name of the parent table 
    
    Parent table is the table whose primary key is used as the foreign key in another table.
    The foreign key relation text is the third item, denoted by index 2, in the list.
    The name is after the word 'REFERENCES' and before the opening parenthesis. 
    
    Args:
        foreign_key_relation_list (list): A list containing the foreign key relation text, among others.
    
    Returns:
        str: The name of the parent table
    """

    start_str = 'REFERENCES'
    end_str = '('
    return extract_substring_between_strings(foreign_key_relation_list[2], start_str, end_str).strip()

def _get_edges(foreign_key_relation_list):
    """ 
    Get the nodes representing the two ends of an edge

    The first item of the tuple is the parent node (i.e., the table whose primary key is used as the foreign key in another table).
    The second item of the tuple is the child node (i.e., the table that is using the primary key of another table as the foreign key).

    Args:
        foreign_key_relation_list (list): A list containing the foreign key relation text, among others.

    Returns:
        tuple: A tuple representing the nodes at the two ends of an edge    
    """

    child_node = foreign_key_relation_list[0]
    parent_node = _get_parent_table(foreign_key_relation_list)
    return (parent_node, child_node)

def get_nodes_and_edges_from_db(conn, db_type, db_name):
    """ 
    Get the nodes and edges from a database. Nodes represent the tables in the database and edges represent the relationships between the tables.

    Args:
        db_type (str): The type of the databse. Options are postgres and mysql.
        db_name (str): The name of the database from which the nodes and the edges have to extracted.

    Returns:
        list, list: Lists containing nodes and edges in the database.
    """

    with conn:
        cursor = conn.cursor()

        if db_type == 'postgres':
            foreign_key_relation_list = get_all_fks(cursor)
            nodes = [i[0] for i in get_table_names_p(cursor)]
        elif db_type == 'mysql':
            rows = get_table_names_m(conn, cursor, db_name)
            nodes = [i[0] for i in rows]
            foreign_key_relation_list = get_all_foreign_key_relationships(conn, cursor, db_name)

    edges = [_get_edges(i) for i in foreign_key_relation_list]
    return nodes, edges