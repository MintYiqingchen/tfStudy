import numpy as np
import numpy.linalg as linalg
import os, random
import os.path
datadir = '../../machinelearninginaction/Ch05/'
def loadDataSet():
    points =[]; labels = []
    with open(os.path.join(datadir,'testSet.txt')) as fr:
        for line in fr:
            ll = line.strip().split()
            # [1.0, x1, x2]-> w1x1+w2x2+w0=0
            points.append([1.0, float(ll[0]),float(ll[1])]); labels.append(ll[-1])
    return points, labels

# -------------plot--------------------
import matplotlib.pyplot as plt
from matplotlib import animation

def plotFitLine(data, labels, weights):
    xs1 = []; xs0 = []; ys1=[]; ys0=[]
    for i, label in enumerate(labels):
        if label=='0':
            xs0.append(data[i][1]); ys0.append(data[i][2])
        else:
            xs1.append(data[i][1]); ys1.append(data[i][2])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = np.arange(-3,3,0.1)
    y = -(weights[0]+weights[1]*x)/weights[2]

    ax.scatter(xs0, ys0, s=30, c='red', marker='s')
    ax.scatter(xs1, ys1, s=30, c='green')
    line, = ax.plot(x,y)
    plt.xlabel('X1'); plt.xlabel('X2')
    return fig, line
# -------------- animation set ----------------

def plotFitLineAnimation(frame, figinfo):
    x = np.arange(-4,4,0.2)
    y = -(frame[1]*x+frame[0])/frame[2]
    line = figinfo
    line.set_data(x,y)
    return line,


sigmoid = lambda x:1.0/(1+np.exp(-x))
def gradAscend(data, labels):
    datamat = np.array(data, dtype='float32')
    labelmat = np.array(labels, dtype='int32')
    m,n = datamat.shape
    alpha = 0.001
    maxCycles = 10000
    weights = np.ones(n)
    for k in range(maxCycles):
        h = sigmoid(datamat.dot(weights))
        error = (labelmat-h)
        #weights = weights+alpha*(datamat.T) @ (error*h*(1-h))
        weights = weights+alpha*(datamat.T) @ error
        if k%10==0: #用于输出动画，可实际使用时注释掉
            yield weights
    #return weights
def SGD(data, labels, num_epoch=150):
    datamat = np.array(data,dtype='float32')
    labelmat = np.array(labels, dtype='int32')
    m,n =datamat.shape
    weights = np.ones(n)
    for j in range(num_epoch):
        for i in range(m):
            alpha = 4/(1+j+i)+0.01
            randidx = random.randint(0,m-1)
            h=sigmoid(datamat[randidx].dot(weights))
            error = (labelmat[randidx]-h)
            weights = weights + alpha*datamat[randidx].dot(error)
        if j%10==0:
            yield weights

class LogisticClassifier(object):
    def __init__(self):
        self.weights = None
        self.iterlog = None
    def fit(self, data, labels):
        self.iterlog = list(SGD(data, labels))
        self.weights = self.iterlog[-1]

    def predict(self,data):
        data = np.array(data, dtype='float32')
        res = 1 if sigmoid(self.weights.dot(data))>=0.5 else 0
        return res

if __name__=='__main__':
    points, labels = loadDataSet()
    classifier = LogisticClassifier()
    classifier.fit(points,labels)
    fig,line = plotFitLine(points, labels, [1,1,1])
    ani = animation.FuncAnimation(fig, plotFitLineAnimation,\
                                  classifier.iterlog, fargs=(line,),\
                                  repeat=False)
    plt.show()
    #for i in range(len(labels)):
    #    print(classifier.predict(points[i]), labels[i])
