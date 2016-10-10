from math import log
import operator
import treePlotter
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        #print featVec
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    #labelCounts = {'yes': 2, 'no': 3}#
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def createDataSet():

    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]

    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    reDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            reDataSet.append(reducedFeatVec)
    return reDataSet
    # splitDataSet(myDat, 0, 1) >>> [[1, 'yes'], [1, 'yes'], [0, 'no']]#
    # splitDataSet(myDat, 0, 0) >>> [[1, 'no'], [1, 'no']]#

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.key():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet) # 0 #
    bestFeatLabel = labels[bestFeat] # labels = ['no surfacing', 'flippers'] #
    print bestFeatLabel
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    print featValues
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLables = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLables) # recursion #
    return myTree
    # myTree >>> {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}} #

def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0] #'no surfacing'#
    secondDict = inputTree[firstStr] # {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}} #
    featIndex = featLabels.index(firstStr) #index function finds out the first str that matchs the firstStr in the featLabels. here featIndex == 0#
    for key in secondDict.keys(): # 0  1 #
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key], featLabels, testVec) #if dict, go on#
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

#myDat, labels = createDataSet()
# labels >>> ['no surfacing', 'flippers'] #
#myTree = treePlotter.retrieveTree(0)
# myTree >>> {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}} #
#classify(myTree, labels, [1,0]) # no #
#storeTree(myTree, 'classifierStorage.txt')
#grabTree('classifierStorage.txt')

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = createTree(lenses, lensesLabels)
treePlotter.createPlot(lensesTree)
