import json
class Matcher:
    def __init__(self):
        """
        Initializes an instance of Matcher with  with None for documents and an empty dictionary
        for labels.

        Args:
            None

        Returns:
            None

        Example:
            >>> matcher = Matcher()
        
        Note:
            - After initialization, `documents` will be set to None and `labels` will be an empty dictionary.
        """

        self.documents = None
        self.labels = {}
    
    def get_documents_from_dkd(self, dkd_filename):
        """
        Read documents from the Domain Knowledge Definition (DKD) file

        Args:
            dkd_filename (str): The name of the DKD file.
        
        Returns:
            None
        
        Example:
            >>> filename = 'dkd.json'
            >>> matcher = Matcher()
            >>> matcher.get_documents_from_dkd(filename)
        
        Note:
            - The DKD file must be a JSON file that adheres to the standard DKD structure.
        """

        with open(dkd_filename, 'r') as file:
            self.documents = json.load(file)['documents']
