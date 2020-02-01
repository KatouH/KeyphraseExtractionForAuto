from collections import defaultdict
import math
import jieba
import logging
import pickle
import os

jieba.setLogLevel(logging.INFO)
default_stopwords_path = "../../DataSet/stop_words.txt"
default_corpus_path = "../../DataSet/train1.txt"
class TFIDF:
    __stopwords = list()
    __corpus = list()
    __mode = 0
    __docnums = 0
    sw_path = ''
    cp_path = ''
    def __init__(self):
        print("Loading corpus...")
        self.setStopWords(default_stopwords_path)
        self.setCorpus(default_corpus_path)
        print("Loading successfully")
    def dataProcessing(self,sentence):
        word_list = list(jieba.cut(sentence,cut_all=False))
        words = []
        for word in word_list:
            if word not in self.__stopwords and word != '\n':
                words.append(word)
        return words
    def setStopWords(self,path):
        self.__stopwords = []
        self.sw_path = path
        with open(self.sw_path,'r',encoding="utf-8") as f:
            for stopword in f.readlines():
                self.__stopwords.append(stopword.strip())
    def setCorpus(self,path,mode=0):
        self.cp_path = path
        self.__mode = mode
        if mode == 0:
            self.__corpus = list()
            with open(self.cp_path,'r',encoding='utf-8') as f:
                for sentence in f.readlines():
                    words = self.dataProcessing(sentence)
                    self.__corpus.append(words)
                self.__docnums = len(self.__corpus)
        elif mode == 1:
            self.__corpus = defaultdict(float)
            with open(self.cp_path,'r',encoding='utf-8') as f:
                for line in f.readlines():
                    arrline = line.split()
                    self.__corpus[arrline[0]] = float(arrline[1])

    def getTFIDF(self,sentence):
        word_list = self.dataProcessing(sentence)
        word_frequecny = defaultdict(int)
        for word in word_list:
            word_frequecny[word] += 1
        total = sum(word_frequecny.values())

        word_idf = {}
        word_doc = defaultdict(int)
        if self.__mode == 0:
            for word in word_frequecny:
                for doc in self.__corpus:
                    if word in doc:
                        word_doc[word] += 1
                word_idf[word] = math.log(self.__docnums/(word_doc[word]+1))
        elif self.__mode == 1:
            for word in word_frequecny:
                word_idf[word] = self.__corpus[word]
        
        word_tf_idf = {}
        for word in word_frequecny:
            word_tf_idf[word] = word_frequecny[word]*word_idf[word]/total
        
        return word_tf_idf

def saveModel(obj,outfile="../../Model/tfidf.pkl"):
    with open(outfile,'w') as f:
        pickle.dump(obj,f)

def loadModel(infile="../../Model/tfidf.pkl"):
    if os.path.exists(infile):
        with open(infile,'r') as f:
            return pickle.load(f)
    else:
        return False

#testing 
if __name__ == "__main__":
    tfidf = TFIDF()
    print(tfidf.getTFIDF("备胎硬伤"))
    tfidf.setCorpus("../../DataSet/idf.txt",1)
    print(tfidf.getTFIDF("备胎硬伤"))
    #print(tf_idf(comment_list[0]))