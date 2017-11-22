import numpy as np
import operator
import matplotlib.pyplot as plt

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
    

def file2matrix(filename):
    ''' 示例数据为datingTestSet.txt'''
    raw_data = []
    with open(filename,encoding='utf8') as f:
        raw_data =np.array([x.split('\t') for x in f ])
    labels = raw_data[:,-1]
    fig = plt.figure()
    resArray = np.array(raw_data[:,:-1],dtype='float64')
    return resArray, labels
def datingClassTest():
    hoRatio = 0.90
    X,Y = file2matrix('./datingTestSet.txt')
    m = len(X)
    trainData, trainLables = autoNorm(X[:int(m*hoRatio)]), Y[:int(m*hoRatio)]
    testData, testLabels = autoNorm(X[int(m*hoRatio):]), Y[int(m*hoRatio):]
    errorCount = 0.0
    for i in range(len(testData)):
        clsRes = classify0(testData[i,:],trainData,trainLables,3)
        print("the classifier result came back with %s, the real answer is %s"%\
                (clsRes, testLabels[i]))
        if clsRes!=testLabels[i]:
            errorCount+=1.0

    print("the total error rate is %f"%(errorCount/float(len(testData))))
def main():
    X, Y = file2matrix('./datingTestSet.txt')
    # res = classify0([7000, 15.5, 0],X,Y,20)
    X = autoNorm(X)
    style = [ord(y[0]) for y in Y]
    ax = plt.subplot(311)
    ax.scatter(X[:,1], X[:,2],c=15.0*np.array(style))
    ax = plt.subplot(312)
    ax.scatter(X[:,0], X[:,1],c=15.0*np.array(style))
    ax = plt.subplot(313)
    ax.scatter(X[:,0],X[:,2],c=15.0*np.array(style))
    plt.show()
if __name__=="__main__":
    # main()
    datingClassTest()
