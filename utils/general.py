import os, re

def make_subdirectory(subdirectory_name):
    """
    Create a new sub directory with the specified name within the current directory.

    Args:
    subdirectory_name (str): The name of the subdirectory to be created.

    Returns:
        bool: True if a sub directory is successfully created, Otherwise, False.
    """

    if not os.path.exists(subdirectory_name):
        os.makedirs(subdirectory_name)
        return True
    else:
        return False

def reset_community_id_numbers(partition, count=0):
    """
    Reset the community ID in a partition dictionary as the algorithm to move connector nodes assigns higher ID values.
    
    Args:
        partition (dict): A dictionary representing a partition where keys are nodes and values are community IDs.
    
    Returns:
        dict: A dictionary representing a partition where the community IDs have been reset and starts incrementally from 0.

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
    Extract the substring located between two specified strings in the given text.

    Args:
        text (str): The input text from which the word will be extracted.
        start_str (str): The starting string marking the beginning of the target word.
        end_str (str): The ending string marking the end of the target word.

    Returns:
        str or None: The substring between start_str and end_str if found, or None if not found.
    
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
    Check if a file with a given filename is in a sub directory within the current direction,

    Args:
        subdirectory_name (str): The name of the sub directory.
        filename (str): The name of the file that needs to be checked in the sub directory.
    
    Returns:
        bool: True if the file is in the sub directory, Otherwise, False.
    
    Example:
        >>> is_file_in_subdirectory('subdir', 'file.txt')
        True
    """

    file_path = os.path.join(os.getcwd(), subdirectory_name, filename)
    return os.path.isfile(file_path)

def read_lines(filename):
    """
    Read lines from a text file and return them as a list.

    Args:
        filename (str): The path to the text file.

    Returns:
        list: A list containing the lines read from the text file.

    Example:
        >>> lines = read_lines('example.txt')
    """

    with open(filename, 'r') as file:
        return [line.strip() for line in file.readlines()]

def filter_values_by_dictionary(values, dictionary):
    """
    Filter a list of values by checking for their presence in the keys of a dictionary.

    Args:
        values (list): The list of values to be filtered.
        dictionary (dict): The dictionary containing keys to be used for filtering.
    
    Returns:
        list: A list containing values from the input list that are also present as keys in the dictionary.

    Example:
    >>> my_dict = {'a': 1, 'b': 2, 'c': 3}
    >>> my_values = ['a', 'b', 'd', 'e']
    >>> result = filter_values_by_dictionary(my_dict, my_values)
    >>> print(result)
    ['a', 'b']
    """

    return list(set(values) & set(dictionary.keys()))