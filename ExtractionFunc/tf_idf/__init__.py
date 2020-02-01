from __future__ import absolute_import
from .tf_idf import TFIDF
import pickle
import os


default_tfidf = TFIDF()

tfidf_extraction = default_tfidf.getTFIDF

def set_corpus(path,mode=0):
    default_tfidf.setCorpus(path,mode)
def set_stopwords(path):
    default_tfidf.setStopWords(path)