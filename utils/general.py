import os
import re
from collections import Counter


def get_word_between_strings(text, start_str, end_str):
    """Extract the word located between two specified strings in the given text.

    Args:
        text (str): The input text from which the word will be extracted.
        start_str (str): The starting string marking the beginning of the target word.
        end_str (str): The ending string marking the end of the target word.

    Returns:
        str or None: The word between start_str and end_str if found, or None if not found.
    """

    pattern = re.compile(rf"{re.escape(start_str)}(.*?){re.escape(end_str)}")
    match = pattern.search(text)
    if match:
        return match.group(1)
    return None


def make_subdirectory(subdirectory_name):
    """Create a new sub directory with the specified name within the current directory.

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