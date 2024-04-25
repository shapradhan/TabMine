import os
import re

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

def reset_community_id_numbers(partition):
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
    count = 0

    for key, value in partition.items():
        if value not in counts_dict:
            counts_dict[value] = count
            count += 1
        fixed_partition[key] = counts_dict[value]
    
    return fixed_partition

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
    
def contains_value(list_of_tuples, value):
    """
    Check if any tuple in the list contains the specified value.

    Args:
        list_of_tuples (list of tuples): The list of tuples to search.
        value: The value to search for within the tuples.

    Returns:
        bool: True if the value is found in any tuple, False otherwise.
    """

    return any(value in a for a in list_of_tuples)

def find_keys_with_value(dictionary, target_value):
    """
    Find keys in a dictionary that have a specified value.

    Args:
        dictionary (dict): The dictionary to search.
        target_value: The value to search for within the dictionary.

    Returns:
        list: A list of keys in the dictionary that have the specified value.
    """

    return [key for key, value in dictionary.items() if value == target_value]

def append_unique_list(main_list, new_list):
    """
    Append elements from a new list to a main list if they are not already present.

    Args:
        main_list (list): The main list to which unique elements will be appended.
        new_list (list): The list containing elements to be appended to the main list.

    Returns:
        None: The function modifies the main list in place.

    Example:
        main_list = [1, 2, 3]
        new_list = [3, 4, 5]
        append_unique_list(main_list, new_list)
        print(main_list)  # Output: [1, 2, 3, 4, 5]
    """

    # Convert the lists to sets for comparison
    main_set = {tuple(sublist) for sublist in main_list}
    new_set = set(new_list)
    
    # If the new_list is not already in the main_list, append it
    if tuple(new_set) not in main_set:
        main_list.append(list(new_set))
    return main_list