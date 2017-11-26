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

def getEntropyDiscrete(label):
    '''
    返回当前数据集的信息熵
    '''
    labelCounter={}
    for ll in label:
        labelCounter[ll] = labelCounter.get(ll, 0)+1
    
    pp = np.array(list(labelCounter.values()),dtype='float64')/len(label)
    etp = -np.sum(pp*np.log2(pp))
    return etp

def splitByFeature(X, Y, i):
    '''根据数据集X的第i个feature划分'''
    resData = {}
    for idx, data in enumerate(X):
        if not resData.get(data[i]):
            Xset = [data]
            Yset = [Y[idx]]
            resData[data[i]] = (Xset, Yset)
        else:
            resData[data[i]][0].append(data)
            resData[data[i]][1].append(Y[idx])
    return resData



class DecisionTree(object):
    def __init__(self, maxHeight=9999):
        self.maxHeight = maxHeight
        self.tree = None
        self.features = None # 存储特征的index
        self.feaname = None # 存储特征的名字
        self.classes = None # 存储总共有哪些分类
    def reset(self):
        self.tree = None
        self.features = None # 存储特征的index
        self.feaname = None # 存储特征的名字
        self.classes = None # 存储总共有哪些分
    def __str__(self):
        return str(self.tree)
    def fit(self, X, Y, feaName):
        self.reset()
        self.tree = {}
        self.features = []
        self.feaname = feaName
        self.maxHeight = min(self.maxHeight, len(self.feaname))
        data = X[:]
        labels = Y[:]
        # print(data, labels)
        self.classes = set(labels)
        self.tree = self.buildTreeRecursive(data, labels, 0, feaMask=[1]*len(X[0]))
        
    def countMajor(self, labels):
        res = {}
        for l in labels:
            res[l] = res.get(l, 0)+1
        major = max(res, key = res.get)
        return major
    
    def buildTreeRecursive(self, data, labels, currHeight, feaMask):
        '''逐层递归建立节点，每个节点是一个字典'''
        currHeight += 1
        # print(currHeight, data, labels)
        if currHeight > self.maxHeight :
            return self.countMajor(labels) # 到达深度直接返回分类
        if labels.count(labels[0]) == len(labels):
            return labels[0] # 全部一样直接返回分类
        
        # 根据划分后信息熵最小的特征进行划分
        minetp = 1e10
        idxBest = None
        nextData = None # 储存最终划分后的集合
        for i in range(len(self.feaname)):
            if feaMask[i]:
                splitedData = splitByFeature(data, labels, i)
                etp = 0.0
                for fv, spd in splitedData.items(): # spd: (X, Y)
                    etp += getEntropyDiscrete(spd[1]) # 只传入标签，计算每个子集信息熵总和
                
                if etp <= minetp: # 若当前信息熵最小
                    idxBest = i
                    nextData = splitedData
                    minetp = etp

        print("the best feature is ",idxBest)
        feaMask[idxBest] = 0
        inode = {}
        subroot = {self.feaname[idxBest]: inode}
        for fv, spd in nextData.items():
            inode[fv] = self.buildTreeRecursive(spd[0], spd[1], currHeight, feaMask)
        feaMask[idxBest] = 1
        return subroot

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
