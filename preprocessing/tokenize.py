#в данном модуле содержится функция токенизации текста

import nltk
import string
from nltk.corpus import stopwords
from preprocessing.morph import morpholize

def GetTokens(text):
    stop_words = stopwords.words("russian")
    tokens = nltk.word_tokenize(text.lower())
    tokens = [i for i in tokens if (i not in string.punctuation) and (i not in stop_words)]
    morphd = []
    for token in tokens:
        nfword = morpholize(token)
        morphd.append(nfword)
    return morphd