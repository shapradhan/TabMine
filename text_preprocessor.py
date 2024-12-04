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
    
    def remove_stopwords(self):
        """
        Remove stopwords from the text attribute of the instance.

        This method processes the text using spaCy to tokenize it, then removes
        stopwords based on spaCy's predefined list. The resulting text is updated
        in the instance's text attribute with stopwords removed.

        Returns:
            self: The instance itself with the text attribute updated.
        """
       
        stop_words = set(nlp.Defaults.stop_words)
        doc = nlp(self.text)
        self.text = ' '.join(token.text for token in doc if token.text.lower() not in stop_words)
        return self
    
    def lemmatize(self):
        """
        Lemmatize the text attribute of the instance.

        This method processes the text using spaCy to tokenize it and then converts
        each token to its base or dictionary form (lemma). The lemmatized tokens 
        are joined back into a single string and updated in the instance's text 
        attribute.

        Returns:
            self: The instance itself with the text attribute updated with lemmatized text.
        """

        doc = nlp(self.text)
        self.text = ' '.join(token.lemma_ for token in doc)
        return self

    def preprocess(self, preprocessing_options, common_terms=None, pos_tagged=False, nouns_only=False):
        """
        Preprocesses the text attribute of the instance based on the provided options.

        This method performs various preprocessing steps on the text, including 
        removing common terms, punctuation, and extra spaces, as well as 
        removing stopwords and lemmatizing the text. The exact preprocessing 
        steps are determined by the `preprocessing_options` dictionary.

        Args:
            preprocessing_options (dict): A dictionary specifying which preprocessing 
                                        steps to apply. The dictionary should include 
                                        the following keys:
                - 'punctuations' (bool): Whether to remove punctuation. Defaults to False.
                - 'stopwords' (bool): Whether to remove stopwords. Defaults to False.
                - 'lemmatizing' (bool): Whether to perform lemmatization. Defaults to False.
            common_terms (set, optional): A set of common terms to remove from the text.
            pos_tagged (bool, optional): If True, returns part-of-speech-tagged output. Defaults to False.
            nouns_only (bool, optional): If True, filters the text to include only nouns. Defaults to False.

        Returns:
            str: The preprocessed text after applying the specified preprocessing steps.

        Example:
            >>> text_preprocessor = TextPreprocessor("The quick brown foxes are jumping over the lazy dogs")
            >>> preprocessing_options = {'punctuations': True, 'stopwords': True, 'lemmatizing': True}
            >>> cleaned_text = text_preprocessor.preprocess(preprocessing_options, common_terms={'foxes', 'dogs'})
            >>> print(cleaned_text)
            quick brown jump lazy dog
        """
        
        # Remove common terms if provided
        if common_terms:
            self.remove_common_terms(common_terms)

        # Remove punctuation if specified in the options
        if preprocessing_options.get('punctuations', False):
            self.remove_punctuation()
        
        # Remove extra spaces
        self.remove_extra_spaces()

        # Remove stopwords if specified in the options
        if preprocessing_options.get('stopwords', False):
            self.remove_stopwords()

        # Lemmatize the text if specified in the options
        if preprocessing_options.get('lemmatizing', False):
            self.lemmatize()

        return self.text
         
        # return self.remove_punctuation().remove_extra_spaces().lemmatize_and_remove_stop_words(pos_tagged, nouns_only).text()
