from __future__ import absolute_import
from .tf_idf import TFIDF
from .tf_idf import loadModel


default_tfidf = loadModel()

tfidf_extraction = default_tfidf.getTFIDF

def set_corpus(path,mode=1):
    default_tfidf.setCorpus(path,mode)
def set_stopwords(path):
    default_tfidf.setStopWords(path)