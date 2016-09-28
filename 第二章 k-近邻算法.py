from numpy import *
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt

def classify0(inX, dataSet, labels, k): #计算距离 分类
    dataSetSize = dataSet.shape[0]  #读取矩阵的长度，shape[0]就是读取矩阵第一维度的长度。dataSetSize = 4
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet #输入[0， 0]与其的差
    sqDiffMat = diffMat ** 2 #差值的各元素平方
    #print sqDiffMat
    sqDistances = sqDiffMat.sum(axis = 1) #平方和. 按行相加，这里sqDistances = [ 2.21  2.    0.    0.01]
    distances = sqDistances ** 0.5 #开方就是距离，欧式距离 [ 1.48660687  1.41421356  0.          0.1       ]
    sortedDistIndicies = distances.argsort() #argsort函数返回的是数组值从小到大的索引值[2 3 1 0],http://blog.csdn.net/maoersong/article/details/21875705
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1 #字典操作classCount = {'A': 1, 'B': 2}
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True) #排序 反序
    #sortedClassCount = [('B', 2), ('A', 1)]
    return sortedClassCount[0][0]

def file2matrix(filename): #文件预处理 使其符合使用要求
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip() #去掉回车符
        listFromLine = line.split('\t') #使用tab字符将整行数据分割成一个元素列表
        returnMat[index, :] = listFromLine[0:3]#前3个元素作为一个list
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet): #归一化
    minVals = dataSet.min(0) #返回最小值 1*3 为[ 0.        0.        0.001156]
    #print minVals
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals #[  9.12730000e+04   2.09193490e+01   1.69436100e+00]
    normDataSet = zeros(shape(dataSet))
    #print normDataSet
    m = dataSet.shape[0] # m=1000
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest(): #分类测试
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #print normMat
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print normMat[numTestVecs:m, :], datingLabels[numTestVecs:m]
        #print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]):
             errorCount += 1.0
    print "the totle error rate is: %f" % (errorCount/float(numTestVecs))

def classifyPerson():
    resultList = ['not at all','in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVales = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVales)/ranges, normMat, datingLabels, 3)
    print "You will probably like this person: ", resultList[classifierResult - 1]

def img2vector(filename):
    returnVect = zeros((1, 1024)) #多维数组
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i + j] = int(lineStr[j]) #将lineStr每行32位字符存入returnVect
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits') #列目录
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))   #m维数组            
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]          # 9_45.txt 分割字符串 9_45
        classNumStr = int (fileStr.split('_')[0])    # 9
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr) #训练矩阵 每行存一个文本数据
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr) #测试数据集
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if(classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))



#classifyPerson()
#datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))
#plt.show()
#normMat, ranges, minVals = autoNorm(datingDataMat)
#print normMat, ranges, minVals
