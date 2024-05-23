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
    
    def _load_or_create_embeddings(self, text, dir, embeddings_dict, model, use_openai):
        """
        Load the embeddings if they exists; otherwise, create embeddings.

        Args:
            text (str): The text for which the embedding has to be created.
            dir (str): The path of the directory in which the embeddings may exists.
            embeddings_dict (dict): An empty dictionary.
            model (str or SentenceTransformer): The name of the model if OpenAI is to be used or a SentenceTransformer model.
            use_openai (bool): A flag indicating whether to use the OpenAI API to generate the embeddings. 
                If True, the embeddings will be generated using OpenAI's model. If False, an alternative embedding method will be used. 
        
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
            embeddings = embedder.create_embeddings(use_openai)
            embeddings_dict[text] = embeddings
            embedder.save_embeddings(embeddings, dir, embeddings_filename)
            
        return embeddings_dict

    def _get_embeddings(self, model, use_openai):
        """
        Get embeddings for the documents from Domain Knowledge Definition file and the labels assigned to the communities.

        Args:
            model (str or SentenceTransformer): The name of the model if OpenAI is to be used or a SentenceTransformer model.
            use_openai (bool): A flag indicating whether to use the OpenAI API to generate the embeddings. 
                If True, the embeddings will be generated using OpenAI's model. If False, an alternative embedding method will be used. 
        
        Returns:
            tuple: A tuple consisting of two values.
            - The first value (list): A list of embeddings for the documents from the Domain Knowledge Definition file.
            - The second value (list): A list of embeddings for the community labels.
                
        Note:
            - This method is intended for internal use within the class and may not be directly accessible from outside the class.
        """

        document_embeddings = {}
        label_embeddings = {}

        DOCUMENTS_DIR = getenv('DOCUMENTS_DIR')
        LABELS_DIR = getenv('LABELS_DIR')

        for doc in self.documents:
            self._load_or_create_embeddings(doc, DOCUMENTS_DIR, document_embeddings, model, use_openai)
        
        for label in self.labels.values():
            self._load_or_create_embeddings(label, LABELS_DIR, label_embeddings, model, use_openai)
        
        return document_embeddings, label_embeddings
    

    def compute_similarity_scores(self, model, use_openai):
        """
        Compute similarity scores between the documents from Domain Knowledge Definition file and the labels assigned to the communities.

        Args:
            model (str or SentenceTransformer): The name of the model if OpenAI is to be used or a SentenceTransformer model.
            use_openai (bool): A flag indicating whether to use the OpenAI API to generate the embeddings. 
                If True, the embeddings will be generated using OpenAI's model. If False, an alternative embedding method will be used. 
        
        Returns:
            dict: A dictionary of dictionaries. The keys in the inner dictionary represents the community labels and the values represent the 
                similarity score between that label and the key of the outer dictionary i.e, the documents.
        """
        
        similarity_scores = {}
        document_embeddings, label_embeddings = self._get_embeddings(model, use_openai)

        for doc, doc_emb in document_embeddings.items():
            temp_dict = {}
            for label, label_emb in label_embeddings.items():
                similarity_score = calculate_average_similarity([doc_emb, label_emb])
                temp_dict[label] = similarity_score
            similarity_scores[doc] = temp_dict

        return similarity_scores          

