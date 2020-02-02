from collections import defaultdict
from operator import itemgetter
import numpy as np
import jieba
import os
import logging

jieba.setLogLevel(logging.INFO)

_get_path = lambda path:os.path.normpath(os.path.join(os.getcwd(),os.path.dirname(__file__),path))
default_stopwords_path = _get_path("../../DataSet/stop_words.txt")

class UndirectGraph:
    #不在init函数中的变量为全局？
    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self,s,e,w):
        self.graph[s].append((s,e,w))
        self.graph[e].append((e,s,w))

class TextRank:
    def __init__(self):
        self.coef  = 0.85
        self.times = 10
        self.__stopwords = list()
        self.__g = UndirectGraph()
        self.__ws = defaultdict(float)
        self.__oSum = defaultdict(float)
        self.setStopWords(default_stopwords_path)

    def buildTextRankGraph(self,sentence,offset=5):
        word_list = self.dataProcessing(sentence)
        cm = defaultdict(int)
        for i,word in enumerate(word_list):
            for j in range(i+1,i+offset):
                if j>=len(word_list):
                    break
                cm[(word,word_list[j])]+=1
        for v,w in cm.items():
            self.__g.addEdge(v[0],v[1],w)
    
    def dataProcessing(self,sentence):
        word_list = list(jieba.cut(sentence,cut_all=False))
        words = []
        for word in word_list:
            if word not in self.__stopwords and word != '\n':
                words.append(word)
        return words
    
    def setStopWords(self,path):
        self.__stopwords = list()
        with open(path,'r',encoding='utf-8') as f:
            for stopword in f.readlines():
               self.__stopwords.append(stopword.strip()) 

    def getTextRank(self, sentence, offset = 5, topK = 1,withWeight = False):
        self.buildTextRankGraph(sentence,offset)
        for vertex,edges in self.__g.graph.items():
            self.__ws[vertex] = 0.0
            self.__oSum[vertex] = sum((edge[2] for edge in edges),0.0)
        
        for x in range(self.times):
            for vertex in self.__g.graph:
                temp = 0 
                for edge in self.__g.graph[vertex]:
                    temp += (edge[2]/self.__oSum[edge[1]])*self.__ws[edge[1]]
                self.__ws[vertex] = (1 - self.coef) + self.coef * temp

        for v, w in self.__ws.items():
            self.__ws[v] = sigmoid(w,True)
        if withWeight:
            ws = sorted(self.__ws.items(),key=itemgetter(1),reverse=True)
        else:
            ws = sorted(self.__ws,key=self.__ws.__getitem__,reverse=True)

        return ws[:topK]

def sigmoid(X,useStatus):
    if useStatus:
        return 1.0 / (1 + np.exp(-float(X)))
    else:
        return float(X)

#testing
if __name__ == "__main__":
    tr = TextRank()
    print(tr.getTextRank("要说不满意的话，那就是动力了，1.5自然吸气发动机对这款车有种小马拉大车的感觉。如今天气这么热，上路肯定得开空调，开了后动力明显感觉有些不给力不过空调制冷效果还是不错的。",topK=10,withWeight=True))