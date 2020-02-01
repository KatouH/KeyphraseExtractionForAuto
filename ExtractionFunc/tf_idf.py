from collections import defaultdict
import math
import jieba
import logging
import pickle
import os

jieba.setLogLevel(logging.INFO)

class TFIDF:
    __stopwords = list()
    __corpus = list()
    def __init__(self,stopwords_path="../DataSet/stop_words.txt",corpus_path="../DataSet/train1.txt"):
        self.sw_file = open(stopwords_path,'r',encoding='utf-8')
        self.cp_file = open(corpus_path,'r',encoding='utf-8')
        print("Loading corpus...")
        for stopword in self.sw_file.readlines():
            self.__stopwords.append(stopword.strip())
        for sentence in self.cp_file.readlines():
            words = self.dataProcessing(sentence)
            self.__corpus.append(words)
        self.__docnums = len(self.__corpus)
        print("Loading successfully")
    def dataProcessing(self,sentence):
        word_list = list(jieba.cut(sentence,cut_all=False))
        words = []
        for word in word_list:
            if word not in self.__stopwords and word != '\n':
                words.append(word)
        return words
    def getTFIDF(self,sentence):
        word_list = self.dataProcessing(sentence)
        word_frequecny = defaultdict(int)
        for word in word_list:
            word_frequecny[word] += 1
        word_tf = {}
        for word in word_frequecny:
            word_tf[word] = word_frequecny[word]/sum(word_frequecny.values())

        word_idf = {}
        word_doc = defaultdict(int)
        for word in word_frequecny:
            for doc in self.__corpus:
                if word in doc:
                    word_doc[word] += 1
            word_idf[word] = math.log(self.__docnums/(word_doc[word]+1))
        
        word_tf_idf = {}
        for word in word_frequecny:
            word_tf_idf[word] = word_tf[word]*word_idf[word]
        
        return word_tf_idf

def saveModel(obj,outfile="../Model/tfidf.pkl"):
    with open(outfile,'w') as f:
        pickle.dump(obj,f)

def loadModel(infile="../Model/tfidf.pkl"):
    if os.path.exists(infile):
        with open(infile,'r') as f:
            return pickle.load(f)
    else:
        return False

#testing 
if __name__ == "__main__":
    tfidf = loadModel()
    if not tfidf:
        tfidf = TFIDF()
    print(tfidf.getTFIDF("空调不太凉，应该是小问题。"))
    #print(tf_idf(comment_list[0]))