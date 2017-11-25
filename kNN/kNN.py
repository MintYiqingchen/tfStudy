import numpy as np
import operator
import matplotlib.pyplot as plt

import os
from sklearn.neighbors import KNeighborsClassifier

from functools import reduce

def autoStandarize(data):
    avgs = data.mean(axis=0)
    stds = data.std(axis=0)
    return np.array([(x-avgs)/stds for x in data])
    
def autoNorm(data):
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    return np.array([(x-mins)/(maxs-mins) for x in data])

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels
# ------- k近邻算法 ------------
def classify0( point, data, labels, k):
    if isinstance(point, list):
        point = np.array(point,dtype='float64')
    assert point.shape == data[0].shape, "data dimenion is not compatible"
    assert len(data)==len(labels), "length of data and that of labels are not compatible"
    assert k<=len(labels),"k is larger than data number"
    l2list=[]
    for i,p in enumerate(data):
        l2 =np.sqrt(np.sum((point-p)**2))
        l2list.append((i,l2))

    l2list = sorted(l2list,key=lambda x:x[1])
    classCount = {}
    for i in range(k):
        voteLabel = labels[l2list[i][0]]
        classCount[voteLabel]=classCount.get(voteLabel,0)+1
        
    cls = max(iter(classCount))
    return cls
# ------------------------------
# -------约会对象预测分类 ----------
def file2matrix(filename):
    ''' 示例数据为datingTestSet.txt'''
    raw_data = []
    with open(filename,encoding='utf8') as f:
        raw_data =np.array([x.split('\t') for x in f ])
    labels = raw_data[:,-1]
    resArray = np.array(raw_data[:,:-1],dtype='float64')
    return resArray, labels

def datingClassTest():
    hoRatio = 0.90
    X,Y = file2matrix('datingTestSet2.txt')
    m = len(X)
    trainData, trainLabels = autoNorm(X[:int(m*hoRatio)]), Y[:int(m*hoRatio)]
    testData, testLabels = autoNorm(X[int(m*hoRatio):]), Y[int(m*hoRatio):]
    errorCount = 0.0
    for i in range(len(testData)):
        clsRes = classify0(testData[i,:],trainData,trainLabels,3)
        #print("the classifier result came back with %s, the real answer is %s"%\
        #        (clsRes, testLabels[i]))
        if clsRes!=testLabels[i]:
            errorCount+=1.0

    print("the total error rate is %f"%(errorCount/float(len(testData))))
    clf = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    clf.fit(trainData, trainLabels)
    sklRes = clf.predict(testData)

    errorCount = 0.0
    for i in range(len(testData)):
        if sklRes[i]!=testLabels[i]:
            errorCount+=1.0
    print("the total error rate is %f"%(errorCount/float(len(testData))))

def main():
    X, Y = file2matrix('datingTestSet2.txt')
    # res = classify0([7000, 15.5, 0],X,Y,20)
    X = autoNorm(X)
    style = np.array([ord(y[0]) for y in Y])
    print(style[:10])
    ax = plt.subplot(311)
    ax.scatter(X[:,1], X[:,2],c=10*style)
    ax = plt.subplot(312)
    ax.scatter(X[:,0], X[:,1],c=10*style)
    ax = plt.subplot(313)
    ax.scatter(X[:,0],X[:,2],c=10*style)
    plt.show()

# ---------------------------------

# ----------- 手写数字分类 --------
def parseData(dirname):
	filenames = os.listdir(dirname)
	labels = []
	vecs = []
	for fname in filenames:
		labels.append(fname.split('.')[0].split('_')[0])
		with open(os.path.join(dirname,fname)) as f:
			vec = reduce(lambda x,y:x+y,[l.strip() for l in f])
			assert len(vec)==1024, "length of digital image vector is fault"
			vecs.append(np.array(list(vec),dtype='float64'))
	return np.array(vecs), np.array(labels,dtype='int32')

def digitalTest(traindir, testdir):
    trainData, trainLabels = parseData(traindir)
    testData, testLabels = parseData(testdir)
    clf = KNeighborsClassifier()
    clf.fit(trainData, trainLabels)
    res = clf.predict(testData)
    errorCount = 0.0
    for i in range(len(res)):
        if res[i]!=testLabels[i]:
            errorCount+=1.0

    print("predict precise: %f"%(errorCount/float(len(testData))))

if __name__=="__main__":
    # main()
    # datingClassTest()
    os.chdir('../../machinelearninginaction/Ch02')
    digitalTest('trainingDigits','testDigits')
