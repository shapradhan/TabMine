import os, re, csv

def make_subdirectory(subdirectory_name):
    """
    Create a new sub directory with the specified name within the current directory.

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
    Reset the community ID in a partition dictionary as the algorithm to move connector nodes assigns higher ID values.
    
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
    Extract the substring located between two specified strings in the given text.

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
    Check if a file with a given filename is in a sub directory within the current direction,

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

def read_lines(filename):
    """
    Read lines from a text file and return them as a list.

    Args:
        - filename (str): The path to the text file.

    Returns:
        - list: A list containing the lines read from the text file.

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
    >>> result = filter_values_by_dictionary(my_values, my_dict)
    >>> print(result)
    ['a', 'b']
    """

    return list(set(values) & set(dictionary.keys()))

def to_boolean(value):
    """
    Convert a string or value to a boolean.

    Args:
        - value (str): The value to be converted to a boolean. This value should be a string that can be interpreted as a boolean. Common representations include 'true', 'false', '1', '0', and their variations in different cases.

    Returns:
        - bool: Returns `True` if the input value is not 'false' or '0' (case-insensitive). Otherwise, returns `False`.

    Example:
        >>> to_boolean('true')
        True
        >>> to_boolean('False')
        False
        >>> to_boolean('0')
        False
        >>> to_boolean('1')
        True
        >>> to_boolean('yes')
        True
    """

    return value not in ['false', '0']


def create_csv_from_text(response_text, filename='communities_labels.csv'):
    """
    Create a CSV file from a given text containing community labels.

    Args:
        - response_text (str): The input text containing community and label information.
        - filename (str): The name of the CSV file to be created.
    """

    # Extract community and label using regex
    matches = re.findall(r'#Community (\d+): (\w+)', response_text)

    # Remove duplicates by converting to a set and back to a list
    unique_matches = list(set(matches))

    # Sort by community
    unique_matches.sort(key=lambda x: int(x[0]))

    # Write to CSV
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['community', 'label'])  # Write header
        writer.writerows(unique_matches)  # Write data

    print(f"CSV file '{filename}' has been created successfully.")

def calculate_max_tokens(num_communities, max_words_per_label=2, avg_tokens_per_word=2, buffer=20):
    """
    Calculate the maximum number of tokens required for a response.

    Args:
        - num_communities (int): The number of communities to include in the response.
        - max_words_per_label (int): The maximum number of words in each label (default is 2).
        - avg_tokens_per_word (int): The average number of tokens per word (default is 2).
        - buffer (int): Additional tokens to add as a buffer (default is 20).

    Returns:
        - int: The estimated maximum number of tokens needed.
    """
    # Tokens for static part of each line: "Community X: "
    static_tokens_per_line = len("Community X: ")

    # Tokens for each label
    tokens_per_label = max_words_per_label * avg_tokens_per_word

    # Tokens for each newline character
    newline_tokens = 1

    # Calculate tokens per line
    tokens_per_line = static_tokens_per_line + tokens_per_label + newline_tokens

    # Total tokens for all communities
    total_tokens = tokens_per_line * num_communities

    # Add buffer tokens
    total_tokens += buffer

    return total_tokens