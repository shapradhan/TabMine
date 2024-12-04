import mysql.connector
import os
import psycopg2

from postgres_data_extractor import get_all_fks, get_table_names as get_table_names_p
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
    Retrieve the nodes (tables) and edges (relationships) from a specified database.

    This function connects to a database, identifies the tables (nodes), and extracts the relationships between them (edges). 
    The exact method of extracting relationships depends on the type of database (PostgreSQL or MySQL).

    Args:
        conn (object): A connection object to the database.
        db_type (str): The type of the database. Options are 'postgres' and 'mysql'.
        db_name (str): The name of the database from which the nodes and edges will be extracted.

    Returns:
        list, list: A tuple containing two lists:
            - The first list contains the nodes (tables) in the database.
            - The second list contains the edges (relationships between tables).
    """

    with conn:
        cursor = conn.cursor()
        if db_type == 'postgresql':
            foreign_key_relation_list = get_all_fks(cursor)
            nodes = [i[0] for i in get_table_names_p(cursor)]
        elif db_type == 'mysql':
            rows = get_table_names_m(conn, cursor, db_name)
            nodes = [i[0] for i in rows]
            foreign_key_relation_list = get_all_foreign_key_relationships(conn, cursor, db_name)

    edges = [_get_edges(i) for i in foreign_key_relation_list]
    return nodes, edges

def create_connnection(db_type, database):
    """
    Establish a connection to a specified database.

    This function creates a connection to a database based on the provided database type. 
    It supports various database types and establishes the necessary connection 
    parameters for interacting with the specified database.

    Args:
        db_type (str): The type of the database (e.g., 'postgres', 'mysql', etc.).
        database (str): The name of the database to connect to.

    Returns:
        object: A connection object that can be used to interact with the database.
    
    Raises:
        ValueError: If the db_type is not supported.
        Exception: If an error occurs while establishing the connection.
    """
    
    if db_type == 'mysql':
        db_config_mysql = {
            'host': os.getenv('MYSQL_HOST'),
            'user': os.getenv('MYSQL_USER'),
            'password': os.getenv('MYSQL_PASSWORD'),
            'port': os.getenv('MYSQL_PORT')
        }
        connection = mysql.connector.connect(**db_config_mysql)
    elif db_type == 'postgresql':
        db_config_postgres = {
            'host': os.getenv('POSTGRES_HOST'),
            'user': os.getenv('POSTGRES_USER'),
            'password': os.getenv('POSTGRES_PASSWORD'),
            'port': os.getenv('POSTGRES_PORT'),
            'dbname': 'postgres'
        }
        print('db_config_postgres', db_config_postgres)
        db_config_postgres['dbname'] = database
        connection = psycopg2.connect(**db_config_postgres)
    return connection
