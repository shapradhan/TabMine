import pandas as pd

class TableColumnsManager:
    def __init__(self, sim_scores):
        """
        Initialize the TableFieldsExplorer with a dictionary of similarity scores.

        Args:
            sim_scores (dict): A dictionary where each key is a document identifier 
                               and each value is another dictionary of labels with 
                               their similarity scores.
        """
        self.sim_scores = sim_scores
    
    def find_candidate_communities(self, threshold=None, top_n=None):
        """
        Finds candidate community labels based on similarity scores.

        Args:
            threshold (float, optional): Minimum similarity score for including a label.
            top_n (int, optional): Maximum number of top labels to include per document.

        Returns:
            dict: Dictionary with document IDs as keys and lists of candidate labels as values.
        """
        candidate_community_labels = {}
        
        for biz_doc, sim_score_biz_doc in self.sim_scores.items():
            # Sort labels by score in descending order
            sorted_labels = [label for label, score in 
                             sorted(sim_score_biz_doc.items(), key=lambda x: x[1], reverse=True)
                             if (threshold is None or score >= threshold)]
            
            # Limit by top_n if specified
            candidate_community_labels[biz_doc] = sorted_labels[:top_n] if top_n else sorted_labels

        return candidate_community_labels

    def get_tables_from_csv(self, categories, csv_file):
        """
        Retrieves tables for each category item from a CSV file.

        Args:
            categories (dict): Dictionary of categories with lists of items.
            csv_file (str): Path to the CSV file containing 'label' and 'tables' columns.

        Returns:
            dict: Dictionary with categories and items, each associated with a list of tables.
        """
        df = pd.read_csv(csv_file)
        
        # Store results by category and item
        tables_dict = {}
        for category, items in categories.items():
            category_tables = {}
            for item in items:
                # Filter rows for the current item
                matching_row = df.loc[df['label'] == item, 'tables']
                
                # Convert 'tables' column string to list if present
                if not matching_row.empty:
                    try:
                        tables = matching_row.iloc[0]
                        if isinstance(tables, str):
                            tables = tables.strip("[]").replace("'", "").split(", ")  # Convert string to list safely
                        category_tables[item] = tables
                    except Exception as e:
                        print(f"Error processing tables for item '{item}': {e}")
                        category_tables[item] = None
                else:
                    category_tables[item] = None  # No tables found

            tables_dict[category] = category_tables

        return tables_dict
    
    def get_table_columns(self, connection, tables_dict):
        """
        Fetches fields of each table in a PostgreSQL database.

        Args:
            connection: Database connection object.
            tables_dict (dict): Dictionary with categories and their associated tables.

        Returns:
            dict: Nested dictionary of table fields per category and item.
        """
        fields_dict = {}
        
        try:
            cursor = connection.cursor()

            for category, tables in tables_dict.items():
                fields_dict[category] = {}
                for subcategory, table_list in tables.items():
                    fields_dict[category][subcategory] = {}
                    if table_list:
                        for table in table_list:
                            cursor.execute("""
                                SELECT column_name
                                FROM information_schema.columns
                                WHERE table_schema = 'public' AND table_name = %s;
                            """, (table,))

                            fields_dict[category][subcategory][table] = [
                                row[0].replace('_', ' ') for row in cursor.fetchall()
                            ]
        except Exception as e:
            print(f"Database error: {e}")
        finally:
            # Ensure connection closure
            cursor.close()
            connection.close()

        return fields_dict