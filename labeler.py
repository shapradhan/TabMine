import csv
import os
import pandas as pd
import re

from openai import AzureOpenAI, BadRequestError

from text_embedder import TextEmbedder
from utils.general import dict_to_json_format, is_file_in_subdirectory

class Labeler:
    """
    A class for managing and labeling community nodes and descriptions using Azure OpenAI API.
    """

    def __init__(self, partition, descriptions):
        """
        Initialize the Labeler with a partition, descriptions, and setup for Azure OpenAI API.

        Args:
            partition (dict): A dictionary mapping nodes to their community IDs.
            descriptions (dict): A dictionary mapping nodes to their descriptions.
        """

        self.partition = partition
        self.descriptions = descriptions
        self.grouped_ids = {}
        self.result_dict = {}
        self.labels = {}
        self.openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY2"),  
            api_version="2024-02-01",
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT2")
        )
        self.deployment_name = os.getenv("OPENAI_MODEL_NAME2")
        self.documents = []
    
    def set_documents(self, documents):
        """
        Set the list of documents to be used by the labeler.

        Args:
            documents (list): A list of document data to process.
        """
        self.documents = documents

    def group_nodes_by_community_id(self):
        """
        Group nodes into communities based on their community IDs from the partition.
        Updates the `grouped_ids` attribute with a mapping of community IDs to node lists.
        """

        for node, community_id in self.partition.items():
            # Ensure each community_id gets its own list in grouped_ids
            if community_id not in self.grouped_ids:
                self.grouped_ids[community_id] = []
            
            # Avoid adding the node twice
            if node not in self.grouped_ids[community_id]:
                self.grouped_ids[community_id].append(node)

    def create_dict(self):
        """
        Create a dictionary that maps community IDs to node descriptions.
        Updates the `result_dict` attribute with descriptions of nodes grouped by their community ID.
        """

        for community_id, nodes in sorted(self.grouped_ids.items()):
            self.result_dict[community_id] = {
                node: self.descriptions[node]
                for node in nodes if node in self.descriptions
            }
    
    def generate_labels(self):
        """
        Generate labels for communities by analyzing node descriptions using Azure OpenAI.
        Updates the `labels` attribute with generated labels for each community.
        """

        initial_system_prompt = """
            We have these names and descriptions of the nodes in the given community, respectively. 
            The table names and their descriptions are separated by a colon.
            What term can be assigned to the community that reflects a common theme based on the node names and descriptions? 
        """

        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY2"),  
            api_version="2024-02-01",
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT2")
        )

        labels = []
        deployment_name=os.getenv("OPENAI_MODEL_NAME2")
        
        for community_id, dict in self.result_dict.items():
            final_system_prompt = f"""
                Please use only one or two words for your response.
                Do not use characters such as slash or a period.
                You must not provide the term that have already been assigned to other communties.
                Following list includes the terms that have already been assigned to other communities.
                {list(self.labels.values())}
            """

            json = dict_to_json_format(dict)

            try:
                response = client.chat.completions.create(
                    model=deployment_name,
                    messages=[
                        {"role": "system", "content": initial_system_prompt},
                        {"role": "user", "content": json},
                        {"role": "system", "content": final_system_prompt},
                    ],
                    temperature=0
                )
            
                response_text = response.choices[0].message.content
                response_text = response_text.removesuffix('.')
                response_text = re.sub(r'[^A-Za-z0-9 ]+', '', response_text)
                labels.append(response_text)
                self.labels[community_id] = response_text
            except BadRequestError as e:
                if 'context_length_exceeded' in str(e):
                    print("Error: Message exceeds the model's maximum context length. Try reducing the message size.")
                else:
                    print(f"An error occurred: {e}")
    
    def save_labels(self):
        """
        Save the generated labels to a CSV file named 'labels.csv'.

        The CSV file will include the community ID, tables (nodes), and their corresponding labels.
        """

        data = []

        for community_id, nodes in self.grouped_ids.items():
            try:
                # Try to access the label for the given community_id
                label = self.labels[community_id]
            except KeyError:
                # Handle the case where the label does not exist for the community_id
                print(f"Warning: No label found for community_id {community_id}")
                label = None  # Or you can use 'N/A', or another default value
                        
            row = {
                'id': community_id,
                'tables': nodes,
                'label': label  # Use the label, even if it is None
            }

            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv('labels.csv', index=False)
    
    def create_short_description(self, table, description):
        """
        Generate a shortened description for a table using Azure OpenAI.

        Args:
            table (str): The name of the table.
            description (str): The description of the table.

        Returns:
            str: A shortened description that summarizes the table's content.
        """
        
        initial_system_prompt = """
            We have the following name and description of a table, respectively.
            The name and description are separated by a colon. 
            Shorten the description to only few words that reflects the information saved in this table.
        """

        final_system_prompt = """
                Please use only few words for your response.
                Do not use characters such as slash or a period in your response.
                Please only write what information the table stores.
        """

        response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": initial_system_prompt},
                    {"role": "user", "content": f"{table} : {description}"},
                    {"role": "system", "content": final_system_prompt},
                ],
                temperature=0
            )
        
        response_text = response.choices[0].message.content
        return response_text

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
        with open(labels_filename, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)    # Skip the header row

            for row in reader:
                # Strip whitespace from each element in the row
                row = [element.strip() for element in row]
                
                # Check if row has at least 3 elements
                if len(row) >= 3:
                    community_id = row[0]
                    tables = row[1]
                    label = row[2]
                    self.labels[community_id] = label
                    
                else:
                    print("Row does not have enough elements:", row)
        return self.labels

    def load_or_create_embeddings(self, text, dir, embeddings_dict):
        """
        Load the embeddings if they exists; otherwise, create embeddings.

        Args:
            text (str): The text for which the embedding has to be created.
            dir (str): The path of the directory in which the embeddings may exists.
            embeddings_dict (dict): An empty dictionary.
        
        Returns:
            dict: A dictionary in which the key represents either the document from Domain Knowledge Definition file or the community label and the
                values represent the embeddings associated with that text.
        """
        text = text.strip()
        embeddings_filename = '{0}_embeddings.npy'.format(text.replace(' ', '_'))

        if is_file_in_subdirectory(dir, embeddings_filename):
            embedder = TextEmbedder()
            embeddings_dict[text] = embedder.load_embeddings_from_file(dir, embeddings_filename)
        else:
            embedder = TextEmbedder(text)
            embeddings = embedder.create_embeddings()
            embeddings_dict[text] = embeddings
            embedder.save_embeddings(embeddings, dir, embeddings_filename)
            
        return embeddings_dict
