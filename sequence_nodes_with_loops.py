import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import tensorflow_hub as hub


def create_sentence_embeddings(text_list, embeddings_identifier):
    model_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    model = hub.load(model_url)
    
    print("creating embedding {0}".format(embeddings_identifier))
    embeddings = model(text_list)
    save_embeddings_to_file(embeddings, "embeddings_{0}.npy".format(embeddings_identifier))
    
    return embeddings

def load_embeddings_from_file(filename):
    print('loading {0}'.format(filename))
    embeddings_array = np.load(filename)

    embeddings_list = embeddings_array.tolist()
    return embeddings_list

def calculate_average_embedding(embeddings):
    if len(embeddings) == 0:
        raise ValueError("Input emebddings list is empty.")
    
    embeddings_array = np.array(embeddings)
    average_embedding = np.mean(embeddings_array, axis=0)   # Average embedding along the first axis (axis=0)
    return average_embedding

def calculate_average_similarity(embeddings):
    if len(embeddings) <= 1:
        raise ValueError("Input embeddings list should have at least 2 elements.")

    embeddings_array = np.array(embeddings)
    similarity_matrix = cosine_similarity(embeddings_array)

    # Exclude self-similarity and calculate the average similarity
    n = len(embeddings)
    average_similarity = (np.sum(similarity_matrix) - n) / (n * (n - 1))

    return average_similarity

def is_file_in_current_folder(filename):
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, filename)
    return os.path.isfile(file_path)

def calculate_cosine_similarity(embedding1, embedding2):
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)

    # Reshape the embeddings to be 2D arrays for cosine_similarity function
    if embedding1.ndim == 1:
        embedding1 = embedding1.reshape(1, -1)
    if embedding2.ndim == 1:
        embedding2 = embedding2.reshape(1, -1)

    similarity = cosine_similarity(embedding1, embedding2)

    # Since the input might be 1D, return the similarity value directly
    if similarity.size == 1:
        return similarity[0, 0]

    # If the input is 2D, return the 2D array of similarity values
    return similarity
def move_until_above_threshold(full_node_list, table_description_dict):
    new_list = []
    for node_list in full_node_list:
        node_list_embeddings = create_sentence_embeddings(node_list, embeddings_identifier='node_list')
        node_list_embeddings_average_similarity = calculate_average_similarity(node_list_embeddings)
        sim_threshold = 0.8

        if node_list_embeddings_average_similarity < sim_threshold:
            all_above_threshold = False

            num_nodes = len(node_list)
            cut_off_index = num_nodes // 2
            if num_nodes % 2 == 0:
                left_nodes = node_list[:cut_off_index]
                right_nodes = node_list[cut_off_index:]
            else:
                left_nodes = node_list[:cut_off_index]
                right_nodes = node_list[cut_off_index+1:]
                mid_node = node_list[cut_off_index]
                mid_node_description = table_description_dict[mid_node]
                mid_node_embeddings = create_sentence_embeddings(text_list=[mid_node_description], embeddings_identifier='mid_node')
            left_nodes_descriptions = [table_description_dict[node] for node in left_nodes]
            right_nodes_descriptions = [table_description_dict[node] for node in right_nodes]
            left_nodes_embeddings = create_sentence_embeddings(text_list=left_nodes_descriptions, embeddings_identifier='contrived_2_left_nodes')
            right_nodes_embeddings = create_sentence_embeddings(text_list=right_nodes_descriptions, embeddings_identifier='contrived_2_right_nodes')

            left_nodes_avg_embeddings = calculate_average_embedding(left_nodes_embeddings)
            right_nodes_avg_embeddings = calculate_average_embedding(right_nodes_embeddings)

            if num_nodes % 2 == 0:
                pass
            else:
                mid_left_similarity_score = calculate_cosine_similarity(mid_node_embeddings, left_nodes_avg_embeddings)
                mid_right_similarity_score = calculate_cosine_similarity(mid_node_embeddings, right_nodes_avg_embeddings)

                if mid_left_similarity_score >= mid_right_similarity_score:
                    left_nodes.append(mid_node)
                else:
                    right_nodes.append(mid_node)

                new_list.append(left_nodes)
                new_list.append(right_nodes)
        else:
            if len(new_list) == 0:
                new_list = node_list
            else:
                new_list.append(node_list)
    return new_list
if __name__ == '__main__':
    nlp = spacy.load('en_core_web_lg')

    df = pd.read_csv('contrived_descriptions.csv')
    # Contrived
    full_nodes_list = ['sales_1', 'sales_2', 'sales_flow_1', 'sales_flow_2', 'bill_1', 'bill_2', 'delivery_1']

    full_nodes_desc = [df.loc[df['tables'] == node, 'descriptions'].iloc[0] for node in full_nodes_list]

    dict_data_records = df.to_dict(orient='records')
    table_description_dict = {item['tables']: item['descriptions'] for item in dict_data_records}

    
    # Calculate the average embeddings similarity in the complete sequence
    full_nodes_embeddings = load_or_create_embeddings(filename='contrived_2_full_nodes_embeddings.npy', embeddings_identifier='contrived_2_full_nodes', nodes=full_nodes_desc)
    full_nodes_embeddings_average_similarity = calculate_average_similarity(full_nodes_embeddings)
    sim_threshold = 0.8

    if full_nodes_embeddings_average_similarity < sim_threshold:
        all_above_threshold = False

    while not all_above_threshold:
        full_nodes_list = move_until_above_threshold([full_nodes_list], table_description_dict)
        full_nodes_list = move_until_above_threshold(full_nodes_list, table_description_dict)
