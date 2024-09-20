import csv, json

from os import getenv
from text_embedder import TextEmbedder
from utils.embeddings import calculate_average_similarity
from utils.general import is_file_in_subdirectory

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

    def get_community_labels(self, labels_filename):
        """
        Read community labels from a file

        Args:
            labels_filename (str): The name of the file in which the community labels are stored.

        Returns:
            None
        
        Example:
            >>> filename = 'labels.csv'
            >>> matcher = Matcher()
            >>> matcher.get_community_labels(filename)
        
        Note:
            - The labels must be stored in a CSV file with the format of community ID, label.
        """

        with open(labels_filename, 'r', newline='') as file:
            reader = csv.reader(file)
            next(reader)

            for row in reader:
                community_id = row[0]
                label = row[1]

                self.labels[community_id] = label
        return self.labels
        
    
    def _load_or_create_embeddings(self, text, dir, embeddings_dict, model):
        """
        Load the embeddings if they exists; otherwise, create embeddings.

        Args:
            text (str): The text for which the embedding has to be created.
            dir (str): The path of the directory in which the embeddings may exists.
            embeddings_dict (dict): An empty dictionary.
            model (str or SentenceTransformer): The name of the model if OpenAI is to be used or a SentenceTransformer model.
        
        Returns:
            dict: A dictionary in which the key represents either the document from Domain Knowledge Definition file or the community label and the
                values represent the embeddings associated with that text.
        
        Note:
            - This method is intended for internal use within the class and may not be directly accessible from outside the class.
        """

        text = text.strip()
        embeddings_filename = '{0}_embeddings.npy'.format(text.replace(' ', '_'))

        if is_file_in_subdirectory(dir, embeddings_filename):
            embedder = TextEmbedder()
            embeddings_dict[text] = embedder.load_embeddings_from_file(dir, embeddings_filename)
        else:
            embedder = TextEmbedder(text, model)
            embeddings = embedder.create_embeddings()
            embeddings_dict[text] = embeddings
            embedder.save_embeddings(embeddings, dir, embeddings_filename)
            
        return embeddings_dict

    def _get_embeddings(self, model):
        """
        Get embeddings for the documents from Domain Knowledge Definition file and the labels assigned to the communities.

        Args:
            model (str or SentenceTransformer): The name of the model if OpenAI is to be used or a SentenceTransformer model.
        
        Returns:
            tuple: A tuple consisting of two values.
            - The first value (list): A list of embeddings for the documents from the Domain Knowledge Definition file.
            - The second value (list): A list of embeddings for the community labels.
                
        Note:
            - This method is intended for internal use within the class and may not be directly accessible from outside the class.
        """

        document_embeddings = {}
        label_embeddings = {}

        DOCUMENT_EMBEDDINGS_DIR = getenv('DOCUMENT_EMBEDDINGS_DIR')
        LABEL_EMBEDDINGS_DIR = getenv('LABEL_EMBEDDINGS_DIR')

        for doc in self.documents:
            self._load_or_create_embeddings(doc, DOCUMENT_EMBEDDINGS_DIR, document_embeddings, model)
        
        for label in self.labels.values():
            self._load_or_create_embeddings(label, LABEL_EMBEDDINGS_DIR, label_embeddings, model)
        
        return document_embeddings, label_embeddings
    

    def compute_similarity_scores(self, model, similarity_measure):
        """
        Compute similarity scores between documents from the Domain Knowledge Definition file and labels assigned to the communities.

        Args:
            model (str or SentenceTransformer): The model used for computing similarity. This can be:
                - A string representing the name of the OpenAI model to use.
                - An instance of SentenceTransformer for other similarity computations.
            similarity_measure (function): A function or callable used to calculate the similarity score between documents and community labels. This function should take two inputs (a document and a label) and return a similarity score.

        Returns:
            dict: A dictionary where:
                - The keys are document identifiers (e.g., document names or IDs).
                - The values are dictionaries with community labels as keys and their corresponding similarity scores as values. Each inner dictionary represents the similarity scores between a particular document and each community label.

        Example:
            {
                'doc1': {
                    'labelA': 0.85,
                    'labelB': 0.75
                },
                'doc2': {
                    'labelA': 0.90,
                    'labelB': 0.65
                }
            }
        """
        
        similarity_scores = {}
        document_embeddings, label_embeddings = self._get_embeddings(model)
        for doc, doc_emb in document_embeddings.items():
            temp_dict = {}
            for label, label_emb in label_embeddings.items():
                similarity_score = calculate_average_similarity([doc_emb, label_emb], similarity_measure)
                temp_dict[label] = similarity_score
            similarity_scores[doc] = temp_dict

        return similarity_scores          

