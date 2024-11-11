import numpy as np
import os

from sklearn.metrics.pairwise import cosine_similarity

from text_embedder import TextEmbedder

class TableScorer:
    def __init__(self, biz_docs, fields, graph, field_weight=0.4, profile_weight=0.4, neighbor_weight=0.2):
        """
        A class to score tables based on business documents, field relevance, and graph-based relationships.

        This class calculates a score for a table based on multiple factors, including its relevance to 
        specified business documents, the importance of certain fields, and its connections to neighboring tables 
        within a graph. Weights for these factors can be customized through parameters.

        Args:
            biz_docs (list): A list of business documents to use for scoring.
            fields (list): A list of fields or columns in the table to evaluate.
            graph (object): A graph representing relationships between tables.
            field_weight (float, optional): The weight for field relevance in the score calculation. Default is 0.4.
            profile_weight (float, optional): The weight for business document relevance in the score calculation. Default is 0.4.
            neighbor_weight (float, optional): The weight for table relationships in the graph. Default is 0.2.
        """

        self.biz_docs = biz_docs
        self.fields = fields
        self.graph = graph
        self.field_weight = field_weight
        self.profile_weight = profile_weight
        self.neighbor_weight = neighbor_weight

    def _get_embeddings(self, text, dir, embeddings_dict):
        """
        Retrieves existing embeddings from a dictionary or file, or creates and stores new embeddings.

        This method first checks if embeddings for the provided text are available in the dictionary. 
        If not, it checks if the embeddings file exists in the specified directory. If the file is found, 
        the embeddings are loaded into the dictionary. If no embeddings are found in the dictionary or the
        file, new embeddings are created and saved to both the dictionary and the file.

        Args:
            text (str): The text label for which to retrieve or create embeddings.
            dir (str): Directory where embeddings are stored.
            embeddings_dict (dict): A dictionary to store embeddings for reuse within the current session.

        Returns:
            numpy.ndarray: The embeddings for the given text.
        """

        text = text.strip()
        embeddings_filename = f"{text.replace(' ', '_')}_embeddings.npy"
        embeddings_path = os.path.join(dir, embeddings_filename)

        # Check if embeddings are already in the dictionary
        if text in embeddings_dict:
            return embeddings_dict[text]

        # Load from file if it exists
        if os.path.isfile(embeddings_path):
            embeddings = np.load(embeddings_path)
        else:
            # Generate and save embeddings if file does not exist
            embedder = TextEmbedder(text)  
            embeddings = embedder.create_embeddings()
            np.save(embeddings_path, embeddings)  

        # Store embeddings in the dictionary for reuse
        embeddings_dict[text] = embeddings
        return embeddings

    def _create_average_embedding(self, embeddings):
        """
        Computes the average embedding from a list of embeddings.
        
        Args:
            embeddings (list): A list of embeddings.

        Returns:
            numpy.ndarray: Average embedding as a numpy array.
        """

        if not embeddings:
            return np.zeros_like(embeddings[0])  # Return a zero vector if embeddings list is empty
        return np.mean(embeddings, axis=0)

    def _calculate_average_similarity(self, embeddings):
        """
        Calculates the average cosine similarity for a list of embeddings.

        Args:
            embeddings (list): A list of embeddings.
            
        Returns:
            numpy.ndarray: Average cosine similarity score.
        """

        if len(embeddings) < 2:
            return 1.0  # If there is only one embedding or none, similarity is trivially 1

        # Calculate pairwise cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Extract the upper triangular part of the similarity matrix, excluding the diagonal
        num_elements = len(embeddings)
        upper_triangle_indices = np.triu_indices(num_elements, k=1)
        average_similarity = np.mean(similarity_matrix[upper_triangle_indices])

        return average_similarity
    

    def identify_relevant_tables(self):
        """
        Identifies relevant tables by computing similarity scores between business document profiles and table profiles
        using embedding-based similarity metrics.

        The method performs the following steps:
        1. Calculates average embeddings for each business document and table.
        2. Computes similarity scores based on profile, individual field, and neighboring table similarities.
        3. Scores tables based on a weighted combination of these similarity scores.
        4. Returns a dictionary of relevant tables for each document and a sorted list of final results.

        Returns:
            tuple: A dictionary `relevant_tables` with relevant tables and their scores per document label, and
                a list `final_results_sorted` of tuples containing (document name, table name, similarity score),
                sorted by document name and score.
        """
        
        relevant_tables = {}
        embeddings_cache = {}
        final_results = []
        
        business_document_field_embeddings_dir = os.getenv('BUSINESS_DOCUMENT_FIELD_EMBEDDINGS_DIR')
        table_column_embeddings_dir = os.getenv('TABLE_COLUMN_EMBEDDINGS_DIR')

        for document in self.biz_docs['documents']:
            biz_doc_name = document['name']
            doc_fields = document['fields']
            
            # Cache and calculate document field embeddings
            doc_field_embeddings = [
                embeddings_cache.setdefault(field, self._get_embeddings(field, business_document_field_embeddings_dir, embeddings_cache))
                for field in doc_fields
            ]
            doc_profile_embedding = self._create_average_embedding(doc_field_embeddings)

            if isinstance(self.fields, dict) and biz_doc_name in self.fields:
                relevant_tables[biz_doc_name] = {}

                for label, tables in self.fields[biz_doc_name].items():
                    table_similarities = {}

                    for table_name, t_fields in tables.items():
                        # Cache and calculate table field embeddings
                        table_field_embeddings = [
                            embeddings_cache.setdefault(field, self._get_embeddings(field, table_column_embeddings_dir, embeddings_cache))
                            for field in t_fields
                        ]
                        table_profile_embedding = self._create_average_embedding(table_field_embeddings)

                        # 1. Profile similarity score (document vs table)
                        profile_similarity_score = self._calculate_average_similarity([doc_profile_embedding, table_profile_embedding])

                        # 2. Individual field-to-field similarity score
                        individual_field_similarity = sum(
                            self._calculate_average_similarity([
                                embeddings_cache.setdefault(field, self._get_embeddings(field, business_document_field_embeddings_dir, embeddings_cache)),
                                embeddings_cache.setdefault(col, self._get_embeddings(col, table_column_embeddings_dir, embeddings_cache))
                            ])
                            for field in doc_fields for col in t_fields
                        ) / (len(doc_fields) * len(t_fields)) if doc_fields and t_fields else 0

                        # 3. Neighbor similarity score calculation
                        neighbor_similarity_scores = []
                        if table_name in self.graph:
                            for neighbor in self.graph.neighbors(table_name):
                                if neighbor in tables:
                                    neighbor_fields = tables[neighbor]
                                    neighbor_field_embeddings = [
                                        embeddings_cache.setdefault(field, self._get_embeddings(field, table_column_embeddings_dir, embeddings_cache))
                                        for field in neighbor_fields
                                    ]
                                    neighbor_similarity_score = self._calculate_average_similarity(doc_field_embeddings + neighbor_field_embeddings)
                                    neighbor_similarity_scores.append(neighbor_similarity_score)

                            neighbor_similarity_score = sum(neighbor_similarity_scores) / len(neighbor_similarity_scores) if neighbor_similarity_scores else 0

                        # Combine scores with respective weights
                        combined_score = (
                            self.field_weight * individual_field_similarity +
                            self.profile_weight * profile_similarity_score +
                            self.neighbor_weight * neighbor_similarity_score
                        )
                        table_similarities[table_name] = combined_score

                    # Sort tables by combined score for the label
                    sorted_tables = sorted(table_similarities.items(), key=lambda item: item[1], reverse=True)
                    relevant_tables[biz_doc_name][label] = sorted_tables

                    # Collect final results without labels
                    final_results.extend((biz_doc_name, table_name, score) for table_name, score in sorted_tables)

        # Sort final results by document name and score in descending order
        final_results_sorted = sorted(final_results, key=lambda x: (x[0], x[2]), reverse=True)

        # Output final results for each document
        # for doc_name, table_name, score in final_results_sorted:
        #     print(f"Document: {doc_name}, Table: {table_name}, Similarity Score: {score}")

        return relevant_tables, final_results_sorted