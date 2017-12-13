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

def word2Set(vobList, doc):
    '''将一篇文档转换成wordbag表示形式'''
    bagvec = [0]*len(vobList)
    for word in doc:
        if word in vobList:
            idx = vobList.index(word)
            bagvec[idx] = 1
    return bagvec

class NaiveBayesClassifier(object):
    def __init__(self, transform=word2Set):
        self.clsWordFrequency = {} # p(X|c)
        self.clsFrequency = {} # dict
        self.classes = {} # dict 词，出现词的总数
        self.dictionary = [] # list
        self.transform=word2Set
    def fit(self, docs, labels):
        '''计算：每个类别的词的出现次数，每个词的出现次数，类别出现次数'''
        self.dictionary = createVocabList(docs)

        onehot = [self.transform(self.dictionary, doc) for doc in docs]
        onehot = np.array(onehot)

        for i,doc in enumerate(onehot):
            cls = labels[i]
            clsWordCounter = self.clsWordFrequency[cls]\
                if self.clsWordFrequency.get(cls) is not None\
                else np.array([1]*len(self.dictionary))
            self.clsWordFrequency[cls]=clsWordCounter+doc
            # 类别总词数
            self.classes[cls]=self.classes.get(cls,0)+sum(doc)
            # 类别出现次数
            self.clsFrequency[cls]=self.clsFrequency.get(cls,0)+1

        self.clsFrequency = {k:self.clsFrequency[k]/len(labels)\
                             for k in self.clsFrequency}
        self.clsWordFrequency = {k:self.clsWordFrequency[k]/\
                                    (self.classes[k]+2) for k in self.clsWordFrequency}

    def _predict(self, doc):
        '''返回docs的类别'''
        onehot = np.array(self.transform(self.dictionary, doc))
        logprobs = {k:0 for k in self.classes }

        for cls in logprobs:
            wordf = self.clsWordFrequency[cls]*onehot
            logpxc = 0
            for f in wordf:
                logpxc += np.log(f) if f!=0 else 0
            logprobs[cls] = np.log(self.clsFrequency[cls])+logpxc
        return logprobs

    def predict(self,docs):
        res = []
        for doc in docs:
            res.append(self._predict(doc))
        return res



if __name__ =="__main__":
    docs , clsvec = loadDataSet()
    classifier = NaiveBayesClassifier()
    classifier.fit(docs, clsvec)
    print(classifier.predict(docs))

