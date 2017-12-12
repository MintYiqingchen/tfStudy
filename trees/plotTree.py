import matplotlib.pyplot as plt
from decisionTrees import DecisionTree

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4",fc="0.8")
arrow_args = dict(arrowstyle="<-")

from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['Ariel']
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

def plotNode(nodeText, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeText, xy=parentPt,xycoords="axes fraction",\
            xytext=centerPt, textcoords="axes fraction",\
            va = "center", ha="center", bbox=nodeType, arrowprops=arrow_args)

def createPlot():
    fig = plt.figure(1, facecolor = "white")
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode(u'decisionNode', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode(u'leafNode', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()
def getNumLeafs(tree):
    numLeaves = 0
    firstStr = tree.keys()[0]
    secondDict = tree[firstStr]
    for key in secondDict.keys():
        if isinstance(secondDict[key],dict):
            numLeaves += getNumLeafs(secondDict[key])
        else:
            numLeaves += 1
    return numLeaves
def getTreeDepth(tree):
    maxDepth = 0
    firstStr = tree.keys()[0]
    secondDict = tree[firstStr]
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            thisDepth = 1+getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth=thisDepth
    return maxDepth

if __name__=="__main__":
    createPlot()
