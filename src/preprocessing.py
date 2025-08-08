"""This module assignes for creating preprocessing function.
Proceses of Tokenization, Stemming, Lemmatization, Handling text (Remove HTML Tag, URLs, Emojies and other) are here."""

import re                                          # Import Regular Expression (remove HTML tags)
import string                                      # Import Punctuation
from textblob import TextBlob                      # Import this Library to Handle the Spelling Issue
import nltk
from nltk.corpus import stopwords                  #  NLTK library to remove Stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import emoji                                       # for translating symbol to text
import spacy                                       # for tokenization
import spacy.cli
# spacy.cli.download("en_core_web_sm")              # for  working with spacy, after the first start should pick  # spacy.cli.download("en_core_web_lg")
from nltk.stem.porter import PorterStemmer          # for stemming
# nltk.download('all')                              # for  working with NLTL function, after the first start should pick #nltk.download('all')
from chat_words import chat_word                    # for translate slang of charts to text
from autocorrect import Speller                     # for Spelling Correction
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Choose items for preprocessing: True or False
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()  # Lemmatization


# Function for preprocessing
def preprocessing(
    text,
    lower=True,                  # LoweCasing Text
    remove_html=True,            # Remove HTML Tag
    remove_url=True,             # Remove URLs
    remove_punc=True,            # Remove punctuation
    change_chat=True,            # Handling chat's words to words
    spell_cor=True,              # Spelling Correction
    remove_stopword=True,        # Remove StopWords
    remove_emoji=True,           # Handling Emojies to words
    use_stemm=False,             # Apply Stemming
    use_token=False,             # Apply Tokenization
):
    if lower:                    # LoweCasing Text
        text = text.lower()

    if remove_html:
        pattern_1 = re.compile("<.*?>")      # constant using one regular expression
        text = re.sub(pattern_1, r"", text)  # Remove HTML Tags (changes ('<.*?>') to gap " ")

    if remove_url:
        pattern_2 = re.compile(r"https?://\S+|www\.\S+")   #  Remove URLs from Text or Whole Corpus.
        text = pattern_2.sub(r"", text)

    if remove_punc:
        punc = string.punctuation          # Remove punctuation
        text = text.translate(str.maketrans("", "", punc))

    if change_chat:
        new_text = []                      # changes chat's words to text
        for i in text.split():
            if i.upper() in chat_word:
                new_text.append(chat_word[i.upper()])
            else:
                new_text.append(i)
        text = " ".join(new_text)
        new_text.clear()

    if spell_cor:
        spell = Speller(lang="en")         # Spelling Correction
        text = spell(text)

    if remove_stopword:
        stopword = stopwords.words("english")  # Handling StopWords
        for word in text.split():
            if word in stopword:
                new_text.append("")
            else:
                new_text.append(word)
        pattern_3 = new_text[:]
        text = " ".join(pattern_3)

    if remove_emoji:
        text = emoji.demojize(text)           # Handling Emojies

    if use_stemm:
                                              # Stemming
        text = " ".join([stemmer.stem(word) for word in text.split()])
    else:
        words = nltk.word_tokenize(text)
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]              # Lemmatization
        text = " ".join(lemmatized_words)

    if use_token:
        nlp = spacy.load("en_core_web_sm")    # the English language model 'en_core_web_sm'
        text = nlp(text)                       # cmd:  python -m spacy download en_core_web_sm

    return text
