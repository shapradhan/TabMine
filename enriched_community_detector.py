import spacy
import numpy as np
nlp = spacy.load('en_core_web_lg')

def get_group_embeddings(group_texts):
    group_embeddings = []
    for text in group_texts:
        doc = nlp(text)
        texts_embedding = np.mean([token.vector for token in doc], axis=0)
        group_embeddings.append(texts_embedding)

def get_average_embedding(group_embeddings):
    return np.mean(group_embeddings, axis=0)
