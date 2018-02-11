#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__  = bayes.py 
__author__ = Hughe 
__time__   = 2018-02-11 00:18 

"""
from numpy import *


def loadDataSet():
    """
    创建实验样本

    :return:    词条切分后的文档集合，类别标签集合
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive(侮辱性), 0 not
    return postingList, classVec

def createVocabList(dataSet):
    """
    创建一个包含在所有文档中出现的不重复的词的列表

    :param dataSet:     文档集合
    :return:            文档中出现的词的列表(不重复的)
    """
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    """
    词汇转化成向量表示(词集模型)

    :param vocabList:   词汇表
    :param inputSet:    某个文档
    :return:            文档向量
    """
    #创建一个与词汇表等长的向量，0表示该词汇未出现，1表示文档中出现该词汇
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec

def bagOfWords2VecMN(vocabList, inputSet):
    """
       词汇转化成向量表示(词袋模型)

       :param vocabList:   词汇表
       :param inputSet:    某个文档
       :return:            文档向量
       """
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    """
    朴素贝叶斯分类器的训练函数

    :param trainMatrix:     文档矩阵
    :param trainCategory:   每篇文档的类别标签向量
    :return:                向量0：每个词是非侮辱词的概率
                            向量1：每个词是侮辱词的概率
                            一篇文档是侮辱性文档的概率
    """
    numTrainDocs = len(trainMatrix)                     #行，有多少篇文档
    numWords = len(trainMatrix[0])                      #列，词汇表长度
    pAbusive = sum(trainCategory)/float(numTrainDocs)   #侮辱性文档数目/总文档数目=一篇文档是侮辱性文档的概率
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones() ,长度为词汇表长度的1数组
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):                       #对每一行(即一篇文档)
        if trainCategory[i] == 1:                       #如果标记为侮辱性的，则该词对应的个数+1,文档总词数+1
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:                                           #否则非侮辱性+1,文档总词数+1
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)          #change to log()
    p0Vect = log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    朴素贝叶斯分类函数

    :param vec2Classify:    要分类的向量
    :param p0Vec:           非侮辱性词汇概率表
    :param p1Vec:           侮辱性词汇概率表
    :param pClass1:         一篇文档是侮辱性文档概率
    :return:                分类结果
    """
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    """
    测试数据

    :return: none
    """
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation', 'stupid', 'help']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage','love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)

#------------------------------------------------------#
#使用朴素贝叶斯过滤垃圾邮件
#------------------------------------------------------#
def textParse(bigString):  # input is big string, #output is word list
    """
    切分文本

    :param bigString:   文档字符串
    :return:            词表(有重复，且已经转换成小写)
    """
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = [];
    classList = [];
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    trainingSet = range(50)
    testSet = []  # create test set
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))  #在0~训练集个数之间生成随机数
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error", docList[docIndex]
    print 'the error rate is: ', float(errorCount) / len(testSet)
    # return vocabList,fullText

#--------------------------------------------------------
#使用朴素贝叶斯分类器从个人广告获取区域倾向
#--------------------------------------------------------
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]           #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ',float(errorCount)/len(testSet)
    return vocabList,p0V,p1V


if __name__ == '__main__':
    #testingNB()
    spamTest()