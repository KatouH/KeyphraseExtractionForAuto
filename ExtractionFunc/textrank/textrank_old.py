from collections import defaultdict
import jieba
import sys


class UndirectGraph:
    #不在init函数中的变量为全局？
    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self,s,e,w):
        self.graph[s].append((s,e,w))
        self.graph[e].append((e,s,w))


def textrank(g,d=0.85,times=10):
    ws = defaultdict(float)
    oSum = defaultdict(float)

    for vertex,edges in g.graph.items():
        ws[vertex] = 1.0 / (len(g.graph) or 1.0) #?从初始化
        oSum[vertex] = sum((edge[2] for edge in edges),0.0) # exp for iter_val in iterable list generation exp
    
    for x in range(times):
        for vertex in g.graph:
            temp = 0
            for edge in g.graph[vertex]:
                temp += (edge[2]/oSum[edge[1]])*ws[edge[1]]
            ws[vertex] = (1 - d) + d * temp
   
    (min_rank, max_rank) = (sys.float_info[0], sys.float_info[3])
 
    # 获取权值的最大值和最小值
    for w in ws.values():
        if w < min_rank:
            min_rank = w
        if w > max_rank:
            max_rank = w
 
    # 对权值进行归一化
    for n, w in ws.items():
        # to unify the weights, don't *100.
        ws[n] = (w - min_rank / 10.0) / (max_rank - min_rank / 10.0)
    return ws

def buildTextRankGraph(word_list,offset):
    graph = UndirectGraph()
    cm = defaultdict(int)
    for i,word in enumerate(word_list):
        for j in range(i+1,i+offset):
            if j>=len(word_list):
                break
            cm[(word,word_list[j])]+=1

    for v,w in cm.items():
        graph.addEdge(v[0],v[1],w)
    
    return graph

#testing
if __name__ == "__main__":
    word_list = jieba.cut("备胎是硬伤")
    tmp = list(word_list)
    g = buildTextRankGraph(tmp,5)
    ws = textrank(g)
    print(ws)