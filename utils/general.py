import json, os, re

def make_subdirectory(subdirectory_name):
    """
    Creates a new sub directory with the specified name within the current directory.

    Args:
        - subdirectory_name (str): The name of the subdirectory to be created.

    Returns:
        - bool: True if a sub directory is successfully created, Otherwise, False.
    """

    if not os.path.exists(subdirectory_name):
        os.makedirs(subdirectory_name)
        return True
    else:
        return False

def reset_community_id_numbers(partition, count=0):
    """
    Resets the community ID in a partition dictionary as the algorithm to move connector nodes assigns higher ID values.
    
    Args:
        - partition (dict): A dictionary representing a partition where keys are nodes and values are community IDs.
    
    Returns:
        - dict: A dictionary representing a partition where the community IDs have been reset and starts incrementally from 0.

    Example:
        >>> partition = {'A': 7, 'B': 18, 'C': 7, 'D': 5}
        >>> reset_community_id_numbers(partition)
        >>> print(partition)
        {'A': 1, 'B': 2, 'C': 1, 'D': 3}  # Community ID numbers are reset to consecutive integers starting from 1.
    """

    counts_dict = {}
    fixed_partition = {}

    for key, value in partition.items():
        if value not in counts_dict:
            counts_dict[value] = count
            count += 1
        fixed_partition[key] = counts_dict[value]
    
    return fixed_partition, count

def extract_substring_between_strings(text, start_str, end_str):
    """
    Extracts the substring located between two specified strings in the given text.

    Args:
        - text (str): The input text from which the word will be extracted.
        - start_str (str): The starting string marking the beginning of the target word.
        - end_str (str): The ending string marking the end of the target word.

    Returns:
        - str or None: The substring between start_str and end_str if found, or None if not found.
    
    Example:
        >>> get_substring_between_strings('gobbledygook', 'ob', 'go')
        'bledy'
    """

    pattern = re.compile(rf"{re.escape(start_str)}(.*?){re.escape(end_str)}")
    match = pattern.search(text)
    if match:
        return match.group(1)
    return None

def is_file_in_subdirectory(subdirectory_name, filename):
    """
    Checks if a file with a given filename is in a sub directory within the current direction,

    Args:
        - subdirectory_name (str): The name of the sub directory.
        - filename (str): The name of the file that needs to be checked in the sub directory.
    
    Returns:
        - bool: True if the file is in the sub directory, Otherwise, False.
    
    Example:
        >>> is_file_in_subdirectory('subdir', 'file.txt')
        True
    """

    file_path = os.path.join(os.getcwd(), subdirectory_name, filename)
    return os.path.isfile(file_path)

def include_nodes_and_edges(nodes, edges, include_nodes):
    """
    Filters the nodes and edges to include only the specified nodes and their associated edges.
    
    Args:
        nodes (list): List of all nodes.
        edges (list of tuple): List of edges, where each edge is a tuple (node1, node2).
        include_nodes (list): List of nodes to include.
    
    Returns:
        tuple: A tuple containing two elements:
            - filtered_nodes (list): List of nodes included in include_nodes.
            - filtered_edges (list of tuple): List of edges where both nodes are in include_nodes.
    """
    
    include_set = set(include_nodes)  # Convert to set for faster lookup

    # Filter nodes
    filtered_nodes = [node for node in nodes if node in include_set]
    
    # Filter edges where both nodes are in include_set
    filtered_edges = [
        edge for edge in edges if edge[0] in include_set and edge[1] in include_set
    ]
    
    return filtered_nodes, filtered_edges

def dict_to_json_format(input_dict):
    """
    Converts a dictionary into a JSON-formatted string.

    This function takes a dictionary where keys represent names and values represent descriptions,
    and transforms it into a JSON-formatted string of a list of dictionaries. Each dictionary in
    the list contains two keys: 'name' (the original dictionary's key) and 'description' 
    (the corresponding value).

    Args:
        input_dict (dict): A dictionary with string keys and values.

    Returns:
        str: A JSON-formatted string representing a list of dictionaries, each with 'name' 
        and 'description' keys, formatted with an indentation of 4 spaces.
    """

    json_list = [{'name': key, 'description': value} for key, value in input_dict.items()]
    return json.dumps(json_list, indent=4)