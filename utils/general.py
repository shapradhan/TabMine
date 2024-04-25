import os
import re
from collections import Counter

def make_subdirectory(subdirectory_name):
    """Creates a new sub directory with the specified name within the current directory.

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
    Resets the community ID in a partition dictionary as the algorithm to move connector nodes assigns higher ID values.
    
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

def is_file_in_subdirectory(subdirectory_name, filename):
    """Check if a file with a given filename is in a sub directory within the current direction,

    Args:
        subdirectory_name (str): The name of the sub directory.
        filename (str): The name of the file that needs to be checked in the sub directory.
    
    Returns:
        bool: True if the file is in the sub directory, Otherwise, False.
    """
    file_path = os.path.join(os.getcwd(), subdirectory_name, filename)
    return os.path.isfile(file_path)


def create_dict_from_df(df, key_col, value_col):
    """ Create a dictionary from a given DataFrame with given columns as keys and values.

    Args:
        df (pandas.DataFrame): The DataFrame from which a dictionary has to be created.
        key_col (str): The name of the column of the DataFrame that will be used as the dictionary key.
        value_col (str): The name of the column of the DataFrame that will be used as the dictionary value.
    
    Returns:
        dict: A dictionary containing values from the given DataFrame as keys and values.
    """

    dict_data_records = df.to_dict(orient='records')
    return {item[key_col]: item[value_col] for item in dict_data_records}


def get_multiple_occuring_values(lst):
    """ Find and return values that occur more than once in a list.

    Args:
        lst (list): The input list containing values to be analyzed.

    Returns:
        list: A list of values that occur more than once in the input list.

    Example:
        lst = [1, 2, 2, 3, 4, 4, 4, 5]
        Returns [2, 4]
    """
    
    # Count occurrences of each element
    element_counts = Counter(lst)

    # Get values that occur multiple times
    values_with_multiple_occurrences = [value for value, count in element_counts.items() if count > 1]

    return values_with_multiple_occurrences


def get_key_by_value(dictionary, value):
    """ Get the key in the given dictionary based on the specified value.
    Args:
        dictionary (dict): The dictionary to search in.
        value: The value to search for in the dictionary.

    Returns:
        The key corresponding to the given value, if found; otherwise, returns None.
    """
    
    for key, val in dictionary.items():
        if val == value:
            return key
    return None 


def check_value_in_list(value1, value2, lst):
    """ Check the presence of value1 or value2 in the given list.

    Args:
        value1: The first value to search for.
        value2: The second value to search for.
        lst (list): The list to search in.

    Returns:
        str or None: value1 if it is present in the given list or value2 if it is present in the given list.
            None if neither of the given values are present in the given list.
    """

    return value1 if value1 in lst else (value2 if value2 in lst else None)

def find_most_common_words(word_list):
    """ Find the most common words in a given list of words.

    Args:
        word_list (list): A list of words to analyze.

    Returns:
        list: A list of the most common words (can contain multiple words).
    """

    joined_texts = ' '.join(word_list)

    # Tokenize and preprocess the text
    words = re.findall(r'\w+', joined_texts) 

    word_counts = Counter(words)
    most_common_count = word_counts.most_common(1)[0][1]

    # Get all words that have the same count as the most common count
    most_common_words = [word for word, count in word_counts.items() if count == most_common_count]

    return most_common_words