from sklearn.feature_extraction.text import HashingVectorizer

from scipy import sparse
from preprocessing.tokenize import GetTokens

def Vectorize(corpus):
    vector_words = HashingVectorizer(tokenizer=GetTokens, analyzer='word', ngram_range=(1,3))
    vector_char = HashingVectorizer(analyzer='char', ngram_range=(3,6))
    x_words = vector_words.fit_transform(corpus)
    x_char = vector_char.fit_transform(corpus)

    return sparse.hstack([x_words, x_char]), vector_words, vector_char