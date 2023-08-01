import numpy as np
def save_embeddings_to_file(embeddings, filename):
    print("saving embedding {0}".format(filename))
    embeddings_array = np.array(embeddings)
    np.save(filename, embeddings_array)
