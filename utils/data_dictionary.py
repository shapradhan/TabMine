def get_table_descriptions(df, nodes):
    """
    Get descriptions of tables from a dataframe

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        nodes (list): A list of nodes in a group

    Returns:
        dict: A dictionary in which the node (table) is the key and its description is the value.
    """
    
    descriptions = {}
    for node in nodes:
       description = df.loc[df['tables'] == node, 'descriptions'].iloc[0]
       descriptions[node] = description
    return descriptions