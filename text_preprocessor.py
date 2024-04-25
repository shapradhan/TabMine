import re
import spacy

from string import punctuation

nlp = spacy.load('en_core_web_lg')

class TextPreprocessor:
    def __init__(self, text):
        """
        Initializes an instance of TextPreprocessor with the provided text and makes the case lower.

        Args:
            text (str): The text to be processed.

        Example:
            >>> text_preprocessor = TextPreprocessor("Hello, world!")
        """

        self.text = text.lower()
   
    def remove_punctuation(self):
        """
        Removes punctuation marks from the text initialized with the instance.

        Returns:
            TextPreprocessor: The instance of TextPreprocessor with the punctuations removed from the text attribute.

        Example:
            >>> text_preprocessor = TextPreprocessor("Hello, world!")
            >>> cleaned_text = text_preprocessor.remove_punctuation()
            >>> print(cleaned_text)
            hello world
        """

        # Create a translation table containing all punctuation characters
        translator = str.maketrans('', '', punctuation)
        self.text = self.text.translate(translator)
        return self

    def remove_common_terms(self, common_terms):
        """
        Removes common terms from the text initialized with the instance.

        Args:
            common_terms (list): A list of common terms.

        Returns:
            TextPreprocessor: The instance of TextPreprocessor with the common terms removed from the text attribute.

        Example:
            >>> text_preprocessor = TextPreprocessor("Hello, world!")
            >>> cleaned_text = text_preprocessor.remove_common_terms(['world'])
            >>> print(cleaned_text)
            hello
        """

        common_terms_set = set(common_terms)
        self.text= ' '.join(part for part in self.text.split() if part not in common_terms_set)
        return self

    def remove_extra_spaces(self):
        """
        Removes extra spaces from the text initialized with the instance.

        Returns:
            TextPreprocessor: The instance of TextPreprocessor with extra spaces removed from the text attribute.

        Example:
            >>> text_preprocessor = TextPreprocessor("Hello, world! ")
            >>> cleaned_text = text_preprocessor.remove_extra_spaces()
            >>> print(cleaned_text)
            hello world
        """

        self.text = re.sub(r'\s+', ' ', self.text).strip()
        return self

    def lemmatize_and_remove_stop_words(self, pos_tagged=False, nouns_only=False):
        """
        Lemmatizes the text initialized with the instance and removes stop words from it.
        
        Args:
            pos_tagged (bool): A Boolean value indicating whether part of speech-tagged output should be returned or not. Defaults to False.
            nouns_only (bool): A Boolean value indicating whether only nouns in a text should be considered. Defaults to False.

        Returns:
            TextPreprocessor: The instance of TextPreprocessor with the lemmatized version of the text attribute with stop words removed from it.

        Example:
            >>> text_preprocessor = TextPreprocessor("The quick brown foxes are jumping over the lazy dogs")
            >>> cleaned_text = text_preprocessor.lemmatize_and_remove_stop_words(pos_tagged=False, nouns_only=False)
            >>> print(cleaned_text)
            quick brown fox jumping lazy dog
        """

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
        """
        Preprocesses the text initialized with the instance.
        
        Args:
            pos_tagged (bool): A Boolean value indicating whether part of speech-tagged output should be returned or not. Defaults to False.
            nouns_only (bool): A Boolean value indicating whether only nouns in a text should be considered. Defaults to False.

        Returns:
            str: A preprocessed text 

        Example:
            >>> text_preprocessor = TextPreprocessor("The quick brown foxes are jumping over the  lazy dogs")
            >>> cleaned_text = text_preprocessor.preprocess(pos_tagged=False, nouns_only=False)
            >>> print(cleaned_text)
            quick brown fox jumping lazy dog
        """

        if common_terms:
            self.remove_common_terms(common_terms)
        
        return self.remove_punctuation().remove_extra_spaces().lemmatize_and_remove_stop_words(pos_tagged, nouns_only).text