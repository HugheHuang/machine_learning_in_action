#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__  = trees.py 
__author__ = Hughe 
__time__   = 2018-02-04 19:12 

"""



from math import log
import operator

def createDataSet():
    """
    创造数据集

    :return: 数据集
    """
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


def calcShannonEnt(dataSet):
    """
    计算信息熵

    :param dataSet: 数据集
    :return:        信息熵
    """
    numEntries = len(dataSet)
    labelCounts = {}
    #字典      类名标签：个数
    for featVec in dataSet:  # 为所有可能分类创建字典
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    """
    按照指定特征划分数据集

    :param dataSet: 待划分数据集
    :param axis:    划分数据集的特征
    :param value:   划分数据集的特征值
    :return:        划分的数据集
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value: #如果数据的第axis个特征的值等于value，将[0,axis)(axis,末尾]的值拼接
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec) #拼接的结果加入新建的数据集
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的方式划分数据集

    :param dataSet: 待划分的数据集
    :return:        最好划分的特征的索引值
    """
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;
    bestFeature = -1
    #每一列的不同值计算信息增益并求和，得到最大信息增益的索引列值并返回
    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature
        uniqueVals = set(featList)  # get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    return bestFeature  # returns an integer


def majorityCnt(classList):
    """
    相当于投票表决器，返回出现次数最多的类
    :param classList:   分类名称列表
    :return:            最佳分类
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    """

    :param dataSet: 数据集
    :param labels:  标签列表
    :return:        决策树(使用数据字典表示)
    """
    classList = [example[-1] for example in dataSet]    #得到每个类别的数目
    if classList.count(classList[0]) == len(classList): #如果类别完全相同，停止划分
        return classList[0]  # stop splitting when all of the classes are equal
    #如果只剩一个数量(表示遍历完所有特征)返回出现次数最多的类别
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    #以下三行得到列表办函的所有属性值
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    """
    使用决策树的分类函数

    :param inputTree:   决策树
    :param featLabels:  合适的标签集
    :param testVec:     测试向量
    :return:            决策结果
    """

    firstStr = inputTree.keys()[0]          #以列表返回一个字典所有的键，取列表第一个值
    secondDict = inputTree[firstStr]        #取得键所对应的值，是一个字典或者是字符串
    featIndex = featLabels.index(firstStr)  #从标签列表中找出键的第一个匹配项的索引位置，标签字符串转换成所在标签集的索引（也就是找出决策树的分类节点）
    key = testVec[featIndex]                #测试向量的相应分量上的值
    valueOfFeat = secondDict[key]           #该值对应字典中的部分，叶子节点或者是子树
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec) #递归分类
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    """
    序列化存储决策树

    :param inputTree:   需要存储的决策树
    :param filename:    保存的文件名
    :return:            none
    """
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    """
    反序列化加载决策树

    :param filename:    文件名
    :return:            决策树
    """
    import pickle
    fr = open(filename)
    return pickle.load(fr)


if __name__ == '__main__':
    import trees
    print help(trees)