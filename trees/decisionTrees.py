#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2017-11-22 15:32:39
# @Author  : MintYi
# Decision Tree

import os
import numpy as np

def getEntropyDiscrete(data, charName=None):
	'''
	返回当前数据集每一特征的信息熵
	'''
	numChar = data.size()[1]
	counterList = [{} for i in range(numChar)]
	for i in range(len(data)):
		for j, v in enumerate(data[i]):
			counterList[j][v] = counterList[j].get(v, 0)+1
	# 逐个计算熵
	for counter in counterList:
		for k,v in counter.items():
			
