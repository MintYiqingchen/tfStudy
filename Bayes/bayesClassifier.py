import numpy as np

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(docs):
    '''从分好词的doc中，抽取单词集合'''
    vobSet = set()
    for doc in docs:
        vobSet = vobSet|set(doc)
    return list(vobSet)

def word2Bag(vobList, doc):
    '''将一篇文档转换成wordbag表示形式'''
    bagvec = [0]*len(vobList)
    for word in doc:
        if word in vobList:
            idx = vobList.index(word)
            bagvec[idx] = 1
    return bagvec

class NaiveBayesClassifier(object):
    def __init__(self):
        self.wordFrequency = dict() # dict
        self.clsWordFrequency = {} #dict
        self.clsFrequency = {} # dict
        self.classes = [] # list
        self.dictionary = [] # list
    def fit(self, docs, labels):
        '''计算：每个类别的词的出现概率，每个词的出现概率，类别出现概率'''
        self.dictionary = createVocabList(docs)

        for i,doc in enumerate(docs):
            for word in doc:
                self.wordFrequency[word] = self.wordFrequency.get(word, 0 )+1

if __name__ =="__main__":
    docs , clsvec = loadDataSet()
    vobList = createVocabList(docs)
    bagvec = word2Bag(vobList, docs[0])
    print(bagvec)
