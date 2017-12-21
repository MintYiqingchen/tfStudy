import numpy as np
import os, random
import os.path

datadir = '../../machinelearninginaction/Ch06/'
def loadDataSet(fname):
    datamat = []; labels = []
    with open(os.path.join(datadir, fname)) as fr:
        for ll in fr:
            larray = ll.strip().split('\t')
            datamat.append([float(larray[0]), float(larray[1])])
            labels.append(int(larray[-1]))

    return datamat, labels

def selectJrand(i, m):
    j=i
    while(j==i):
        j=random.randint(0,m-1)
    return j

def clipAlpha(a, lo, hi):
    a = a if a<=hi else hi
    a = a if a>=lo else lo
    return a

class SVMClassifier(object):
    def fit(self, data, labels, C=6, toler=0.001, maxIter=40):
        datamat = np.array(data, dtype="float32")
        labelmat = np.array(labels,dtype='int32')
        self.b, self.alphas = self._simpleSMO(datamat, labelmat,C,toler,maxIter)
        self.w = self._getW(self.alphas, labels,datamat)

    def predict(self, data):
        res = []
        for d in data:
            d = np.array(d, dtype='float32')
            p = d.dot(self.w)+self.b
            if p>0: res.append(1)
            else: res.append(-1)
        return res

    def _simpleSMO(self, datamat, labelmat, C, toler, maxIter):
        b = 0;m,n = datamat.shape
        alphas = np.zeros([1,m]).reshape(m)
        iteration = 0
        # update lang coefficiences alpha
        while(iteration<maxIter):
            alphaPairsChanged = 0
            for i in range(m):
                # w = \sum yi*ai*xi
                fXi = datamat[i].dot(datamat.T.dot((alphas*labelmat).T)) + b
                Ei = fXi - float(labelmat[i])
                if((labelmat[i]*Ei<-toler) and (alphas[i]<C)) or \
                        ((labelmat[i]*Ei>toler) and (alphas[i]>0)):
                    # alpha_i doesnt satisfy kkt condition
                    j = selectJrand(i, m)
                    fXj = datamat[j].dot(datamat.T.dot((alphas*labelmat).T)) + b
                    Ej = fXj - labelmat[j]

                    alphaIold = alphas[i].copy()
                    alphaJold = alphas[j].copy()

                    if(labelmat[i]!=labelmat[j]):
                        L=max(0,alphas[j]-alphas[i])
                        H=min(C,C+alphas[j]-alphas[i])
                    else:
                        L=max(0,alphas[j]+alphas[i]-C)
                        H=min(C,alphas[j]+alphas[i])
                    if H==L:
                        print("L==H");continue

                    eta = 2*datamat[i].dot(datamat[j].T)-\
                        datamat[i].dot(datamat[i].T)-\
                        datamat[j].dot(datamat[j].T)
                    if eta>=0:
                        print("eta>0"); continue
                    alphas[j] -= labelmat[j]*(Ei-Ej)/eta
                    alphas[j] = clipAlpha(alphas[j], L, H)
                    if(abs(alphas[j]-alphaJold)<0.00001):
                        print("j not moving enough"); continue
                    alphas[i] += labelmat[j]*labelmat[i]*(alphaJold-alphas[j])
                    b1 = b-Ei-labelmat[i]*(alphas[i]-alphaIold)*\
                        datamat[i].dot(datamat[i].T)-\
                        labelmat[j]*(alphas[j]-alphaJold)*\
                        datamat[i].dot(datamat[j].T)
                    b2 = b-Ej-labelmat[i]*(alphas[i]-alphaIold)*\
                        datamat[i].dot(datamat[j].T)-\
                        labelmat[j]*(alphas[j]-alphaJold)*\
                        datamat[j].dot(datamat[j].T)

                    if(0<alphas[i] and C>alphas[i]): b=b1
                    elif(0<alphas[j] and C>alphas[j]): b=b2
                    else:
                        b=(b1+b2)/2

                    alphaPairsChanged += 1

                    print("iter: %d i:%d, pairs changed:%d"%\
                          (iteration, i, alphaPairsChanged))
                if(alphaPairsChanged==0): iteration+=1
                else: iteration=0
                print("iteration num: %d"%iteration)
            return b,alphas

    def _getW(self, alphas, labels, datamat):
        w = datamat.T.dot(alphas*labels)
        return w

if __name__=="__main__":
    classifier = SVMClassifier()
    data, labels = loadDataSet('testSet.txt')
    classifier.fit(data, labels)
    preds = classifier.predict(data)
