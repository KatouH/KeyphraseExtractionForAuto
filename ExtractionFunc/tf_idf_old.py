from collections import defaultdict
import math
import jieba

stopword_list = []
stopwords_file = open("../DataSet/stop_words.txt","r",encoding="utf-8")
for stopword in stopwords_file.readlines():
    stopword_list.append(stopword.strip())

with open("../DataSet/dataset1.txt",'r',encoding="utf-8") as f:
    comment_list = []
    words = []
    comments = f.readlines()
    for comment in comments:
        word_list = list(jieba.cut(comment,cut_all=False))
        for word in word_list:
            if word not in stopword_list and word != '\n':
                words.append(word)
        comment_list.append(words)
        words=[]
#print(comment_list)

def tf_idf(comment):
    word_frequecny = defaultdict(int)
    for word in comment:
        word_frequecny[word] += 1
    
    word_tf = {}
    for word in word_frequecny:
        word_tf[word] = word_frequecny[word]/sum(word_frequecny.values())
    
    doc_nums = len(comment_list)
    word_idf = {}
    word_doc = defaultdict(int)
    for word in word_frequecny:
        for comment in comment_list:
            if word in comment:
                word_doc[word] += 1
        word_idf[word] = math.log(doc_nums/(word_doc[word]+1))
    
    word_tf_idf = {}
    for word in word_frequecny:
        word_tf_idf[word] = word_tf[word]*word_idf[word] 
    
    return word_tf_idf

if __name__ == "__main__":
    print(tf_idf(comment_list[0]))