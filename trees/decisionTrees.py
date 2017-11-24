#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2017-11-22 15:32:39
# @Author  : MintYi
# Decision Tree

import os
import numpy as np
import itertools
def createData():

    dataSet = [[1,1],[1,1],[1,0],[0,1]]
    labels = ['yes','yes','no','no']
    charName = ['no surfacing', 'flipper']
    return dataSet, labels, charName

def getEntropyDiscrete(data, charName=None):
    '''
    返回当前数据集每一特征的信息熵
    '''
    numChar = len(data[0])
    counterList = [{} for i in range(numChar)]
    for i in range(len(data)):
        for j, v in enumerate(data[i]):
            counterList[j][v] = counterList[j].get(v, 0)+1
    # 逐个计算熵
    etpRes = []
    for counter in counterList:
        pp = np.array([v/len(data) for v in counter.values()])
        etp = np.sum(-pp*np.log2(pp))
        etpRes.append(etp)
    return np.array(etpRes)

def chooseBestFeature(data):
    '''返回信息增益最大的feature'''
    etps = getEntropyDiscrete(data)
    idxBest = etps.argmax()
    print('best feature is:%d'%(idxBest))
    return idxBest

class DecisionTree(object):
    def __init__(self, maxHeight=9999):
        self.maxHeight = maxHeight
        self.tree = None
        self.features = None # 存储特征的index
        self.feaname = None # 存储特征的名字
        self.classes = None # 存储总共有哪些分类

    def fit(self, X, Y, feaName):
        self.tree = {}
        self.features = []
        self.feaname = feaName
        self.maxHeight = min(self.maxHeight, len(self.feaname))
        data = X[:]
        labels = Y[:]
        print(data, labels)
        self.classes = set(labels)
        self.tree = self.buildTreeRecursive(data, labels, 0)
        
    def countMajor(self, labels):
        res = {}
        for l in labels:
            res[l] = res.get(l, 0)+1
        major = max(res, key = res.get)
        return major
    
    def buildTreeRecursive(self, data, labels, currHeight):
        '''逐层递归建立节点'''
        currHeight += 1
        print(currHeight, data, labels)
        if currHeight > self.maxHeight :
            return self.countMajor(labels) # 到达深度直接返回分类
        if labels.count(labels[0]) == len(labels):
            return labels[0] # 全部一样直接返回分类
        idxBest = chooseBestFeature(data) 
        self.features.append(idxBest) 
        subsets = {}
        for i,d in enumerate(data):
            fv = d[idxBest]
            if subsets.get(fv):
                subsets[fv].append((d, labels[i]))
            else :
                subsets[fv] = [(d, labels[i])]
        print(subsets)
        # 划分完毕后每一个类进行递归分类
        subRoot = {}
        for k,v in subsets.items():
            data = [ x[0] for x in v]
            labels = [ x[1] for x in v]
            subRoot[k] = self.buildTreeRecursive(data, labels,currHeight)
        return subRoot
    
    def predict(self, X):
        assert len(X)==len(self.feaname), "the number of feature doesn't match"
        # 从根部开始
        subroot = self.tree
        for idxF in self.features:
            fv = X[idxF]
            subroot = subroot[fv]
            if not isinstance(subroot, dict):
                res = subroot
                break
        return res
        
if __name__=="__main__":
    data, labels, feaName=createData()
    clf = DecisionTree()
    clf.fit(data, labels, feaName)
    print( labels)
    for d in data:
        print("Predict result: ",clf.predict(d))

    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf.fit(data, labels)
    print(clf.predict(data))
