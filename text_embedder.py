import os, numpy as np, re

from openai import AzureOpenAI
from utils.general import make_subdirectory

class TextEmbedder:
    def __init__(self, text=''):
        """
        Initialize the instance with the provided text and model.

        This constructor method sets up an instance of the class with optional text and model parameters.
        The text parameter can be used to provide initial text data, while the model parameter can be used
        to specify a pre-trained model for embeddings generation.

        Args:
            - text (str, optional):  The initial text data to be associated with the instance. The default is an empty string.
        
        Returns:
            None

        Example:
            >>> instance = TextEmbedder(text='Sample text')
            >>> print(instance.text)
            'Sample text'
        """
        self.text = text
   
    def create_embeddings(self):
        """
        Generate embeddings for a given input string using the specified model or embedding method.

        Returns:
            - list[float] or tf.Tensor: A list of floats or a TensorFlow tensor containing the embeddings for the input string. 
                The type depends on the model used for generating the embeddings.

        Example:
            >>> embeddings = self.create_embeddings()
            >>> print(embeddings)
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, ...]
        """

        client = AzureOpenAI(
            api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
            api_version = os.getenv("AZURE_OPENAI_VERSION"),  
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") 
        )      

        response = client.embeddings.create(
            input = self.text,
            model = os.getenv("OPENAI_MODEL_NAME") 
        )

        return response.data[0].embedding


    def save_embeddings(self, embeddings, folder_name, filename):
        """Saves the embeddings to a file in a given folder

        Args:
            - embeddings (tf.Tensor):  A TensorFlow EagerTensor containing the embeddings to be saved.
            - folder_name (str): The name of a folder where the file will be saved
            - filename (str): The name of the file to be created.
        
        Returns:
            None
        """
        
        make_subdirectory(folder_name)
        file_path = os.path.join(folder_name, filename)
        np.save(file_path, embeddings)

    def load_embeddings_from_file(self, subfolder_name, filename):
        """Loads the embeddings of a particular file stored in a subfolder
        
        Args:
            - subfolder_name (str): The name of the subfolder
            - filename (str): The name of the file in which the embeddings are stored

        Returns:
            - numpy.ndarray: an array of embeddings
        """

        file_path = os.path.join(os.getcwd(), subfolder_name, filename)
        embeddings_array = np.load(file_path)
        return embeddings_array