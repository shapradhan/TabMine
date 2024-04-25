import re
import spacy

from string import punctuation

nlp = spacy.load('en_core_web_lg')

class TextPreprocessor:
    def __init__(self, text):
        self.text = text.lower()
   
    def remove_punctuation(self):
        # Create a translation table containing all punctuation characters
        translator = str.maketrans('', '', punctuation)
        self.text = self.text.translate(translator)
        return self

    def remove_common_terms(self, common_terms):
        common_terms_set = set(common_terms)
        self.text= ' '.join(part for part in self.text.split() if part not in common_terms_set)
        return self

    def remove_extra_spaces(self):
        self.text = re.sub(r'\s+', ' ', self.text).strip()
        return self

    def lemmatize_and_remove_stop_words(self, pos_tagged=False, nouns_only=False):
        stop_words = nlp.Defaults.stop_words
        doc = nlp(self.text)
        tokens = []
        
        if nouns_only:
            tokens = [token.lemma_ for token in doc if token.pos_ == 'NOUN']
        else:
            if pos_tagged:
                tokens = [(token.lemma_, token.pos_) for token in doc ]
            else:
                tokens = [token.lemma_ for token in doc]

        self.text = ' '.join(token for token in tokens if token[0] not in stop_words)
        return self 
    
    def preprocess(self, common_terms=None, pos_tagged=False, nouns_only=False):
        if common_terms:
            self.remove_common_terms(common_terms)
        
        return self.remove_punctuation().remove_extra_spaces().lemmatize_and_remove_stop_words(pos_tagged, nouns_only).text
    