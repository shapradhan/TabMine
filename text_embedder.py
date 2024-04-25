import os
import numpy as np
import re
import spacy

from utils.general import make_subdirectory

nlp = spacy.load('en_core_web_lg')

class TextEmbedder:
    def __init__(self, text='', model=None):
        self.text = text
        self.model = model
   
    def create_embeddings(self):
        """Retrieves embeddings for a given input string using the specified embedding model.

        Args:
            input_string (str): The input string for which embeddings are to be generated.
            embedding_model (callable): A callable object representing the embedding model. The model must take
                a list of input strings and return a list of embeddings for those strings.

        Returns: 
            tf.Tensor: A TensorFlow tensor containing the embeddings for the input string.
        """

        self.text = re.split(r'[.!?]', self.text)
        self.text = [s.strip() for s in self.text if s.strip()]
        
        # Compute embeddings for the input sentences
        embeddings = self.model.encode(self.text, convert_to_tensor=True)
        return embeddings

    def save_embeddings(self, embeddings, folder_name, filename):
        """Saves the embeddings to a file in a given folder

        Args:
            embeddings (tf.Tensor):  A TensorFlow EagerTensor containing the embeddings to be saved.
            folder_name (str): The name of a folder where the file will be saved
            filename (str): The name of the file to be created.
        
        Returns:
        
        """
        
        make_subdirectory(folder_name)
        file_path = os.path.join(folder_name, filename)
        np.save(file_path, embeddings)

    def load_embeddings_from_file(self, subfolder_name, filename):
        """Loads the embeddings of a particular file stored in a subfolder
        
        Args:
            subfolder_name (str): The name of the subfolder
            filename (str): The name of the file in which the embeddings are stored

        Returns:
            numpy.ndarray: an array of embeddings
        """

        file_path = os.path.join(os.getcwd(), subfolder_name, filename)
        embeddings_array = np.load(file_path)
        return embeddings_array