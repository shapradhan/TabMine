import os
import re


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

def make_subfolder(subfolder_name):
    """Create a new subfolder with the specified name within the current directory.

    Args:
    subfolder_name (str): The name of the subfolder to be created.

    Returns:
        bool: True if a subfolder is successfully created, False otherwise.
    """

    if not os.path.exists(subfolder_name):
        os.makedirs(subfolder_name)
        return True
    else:
        return False

def is_file_in_subfolder(subfolder_name, filename):
    """Check if a file with a given filename is in a subfolder within the current direction,

    Args:
        subfolder_name (str): The name of the subfolder.
        filename (str): The name of the file that needs to be checked in the subfolder.
    
    Returns:
        bool: True if the file is in the subfolder, False otherwise.
    """
    file_path = os.path.join(os.getcwd(), subfolder_name, filename)
    return os.path.isfile(file_path)


def all_values_higher_than(lst, threshold):
    """Check if all values in a list are higher than a certain threshold.
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

    Returns:
        bool: True if all values are higher than the threshold, False otherwise.
    """
    return all(value > threshold for value in lst) 