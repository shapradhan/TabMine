import numpy as np
import tensorflow_hub as hub

def save_embeddings_to_file(embeddings, filename):
    print("saving embedding {0}".format(filename))
    embeddings_array = np.array(embeddings)
    np.save(filename, embeddings_array)

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
