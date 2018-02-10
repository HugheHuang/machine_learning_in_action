#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__  = kNN.py 
__author__ = Hughe 
__time__   = 2018-02-03 09:01 

"""

from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

"""
分类器
"""
def classify0(inX, dataSet, labels, k):
    #第一维度大小(行维度)
    dataSetSize = dataSet.shape[0]
    #tile(复制, (行复制次数, 列复制次数))
    #此处计算矩阵差
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    #计算差值的平方
    sqDiffMat = diffMat**2
    #差值平方求和
    sqDistances = sqDiffMat.sum(axis=1)
    #计算差值平方求和开根号得出欧式距离
    distances = sqDistances**0.5
    #argsort函数返回的是数组值从小到大的索引值
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 对前k个对应的标签值计数
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #排序对象：classCount.iteritems()
    #排序参数：按照字典的值进行排序(在维度1)
    #降序排序
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
        return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')


"""
将文件转换成矩阵
"""
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    #zeros 返回来一个给定形状和类型的用0填充的数组
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        #取得index行所有元素
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
"""
归一化特征值
"""
def autoNum(dataSet):
    #newValue = (oldValue - min) / (max - min)
    #参数0表示从列中取最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals

"""
分类器针对约会网站测试
"""
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNum(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    #将样本集分为训练集和测试集
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 3)
        print "the classifier came back with: %d, the real answer is : %d" % (classifierResult, datingLabels[i])
        if(classifierResult != datingLabels[i]) : errorCount +=1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))

"""
约会网站测试函数
"""
def classifyPerson():
    resultList = ['not at all', 'in small dosed', 'in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent filter miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNum(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
    print "You will probably like this person: ", resultList[classifierResult - 1]

if __name__ == '__main__':
    pass
