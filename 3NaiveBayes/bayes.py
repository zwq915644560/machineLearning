#!/usr/bin/python
# -*- coding:utf-8 -*-

'a simple naive Bayes classifier.'

__author__='zwq'

import numpy as np 

# prepare data, build word vector.
def loadDataSet():
	postList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
			  ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
			  ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
			  ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
			  ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
			  ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec=[0, 1, 0, 1, 0, 1]
	return postList, classVec

def createVocabList(dataSet):
	vocabSet=set([])
	for post in dataSet:
		vocabSet|=set(post)
	return list(vocabSet)

def setOfWords2Vec(vocabSet, post):
	returnVec=[0]*len(vocabSet)
	for word in post:
		#returnVec[vocabSet.index(word)]=1
		returnVec[vocabSet.index(word)]+=1
	return returnVec


# train the algorithm
def trainNB0(trainMatrix, trainCategory):
	'''
	1. 计算类别1（有辱骂内容）出现的概率p1，则类别2的概率为p0=1-p1

	2. 再分别计算在类别0和1的各自条件下，各单词的出现概率p(W|c0)和p(W|c1)

	3. 最终根据贝叶斯公式p(ci|W)=p(ci,W)/p(W)=[p(ci)*p(W|ci)]/p(W),分别计算出p(c0|W)和p(c1|W)，
	分别表示在W这个词向量的条件下，待预测的对象属于c0和c1的概率，比较两者大小，将对象归为概率较大者的类别。
	
	其中，在p(c0|W)和p(c1|W)的计算中:
		由朴素贝叶斯假设：1.特征之间相互独立；2.每个特征同等重要，p(W|ci)等于p(w1|ci)*p(w1|ci)...p(wn|ci);
		而且，由于二者的p(W)值相同，该值不需要计算。
	'''
	pAbusive=trainCategory.count(1)/float(len(trainCategory))
	#p0Num=np.zeros(len(trainMatrix[0]))
	#p1Num=np.zeros(len(trainMatrix[0]))
	#p0Denom=0.0
	#p1Denom=0.0
	p0Num=np.ones(len(trainMatrix[0]))
	p1Num=np.ones(len(trainMatrix[0]))
	p0Denom=2.0
	p1Denom=2.0
	numPost=len(trainCategory)
	for i in range(numPost):
		if trainCategory[i]==0:
			p0Num+=trainMatrix[i]
			p0Denom+=sum(trainMatrix[i])
		elif trainCategory[i]==1:
			p1Num+=trainMatrix[i]
			p1Denom+=sum(trainMatrix[i])
	#p0Vec为类别0的条件下（无辱骂内容），各单词出现的概率（概率向量），即p(W|c0),W为词向量
	#p0Vec=p0Num/p0Denom
	#p1Vec=p1Num/p1Denom
	p0Vec=np.log(p0Num/p0Denom)
	p1Vec=np.log(p1Num/p1Denom)
	return p0Vec, p1Vec, pAbusive

# naive Bayes classifier
def classifyNB(vec2classify, p0Vec, p1Vec, pClass1):
	objVec0=vec2classify*p0Vec  #（整个样本中）在类别0的条件下，待预测对象的各个单词的出现概率
	p0=objVec0.sum()+np.log(1-pClass1)
	objVec1=vec2classify*p1Vec
	p1=objVec1.sum()+np.log(pClass1)
	return 1 if p1>p0 else 0

# test the classifier
def testingNB():
	postList, classVec=loadDataSet()
	vocabSet=createVocabList(postList)
	trainMatrix=[]
	for post in postList:
		trainMatrix.append(setOfWords2Vec(vocabSet, post))
	p0Vec, p1Vec, pAbusive=trainNB0(trainMatrix, classVec)
	
	testEntry=['love', 'my', 'dalmation']
	testVec=np.array(setOfWords2Vec(vocabSet, testEntry))
	print testEntry, 'classified as:', classifyNB(testVec, p0Vec, p1Vec, pAbusive)

	testEntry=['stupid', 'garbage']
	testVec=np.array(setOfWords2Vec(vocabSet, testEntry))
	print testEntry, 'classified as:', classifyNB(testVec, p0Vec, p1Vec, pAbusive)

def textParse(bigString):
	import re
	listOfTokens=re.split(r'\W*', bigString)
	return [i.lower() for i in listOfTokens if len(i)>2]

def spamTest():
	docList=[]; classList=[]
	for i in range(25):
		wordList=textParse(open('./email/spam/%d.txt' %(i+1)).read())
		docList.append(wordList)
		classList.append(1)
		wordList=textParse(open('./email/ham/%d.txt' %(i+1)).read())
		docList.append(wordList)
		classList.append(0)
	vocabList=createVocabList(docList)
	trainSet=range(50); testSet=[]
	for i in range(10):
		randIndex=int(np.random.uniform(0, len(trainSet)))
		testSet.append(randIndex)
		del(trainSet[randIndex])
	trainMatrix=[]; trainClass=[]
	for docIndex in trainSet:
		trainMatrix.append(setOfWords2Vec(vocabList, docList[docIndex]))
		trainClass.append(classList[docIndex])
	p0V, p1V, pSpam=trainNB0(trainMatrix, trainClass)
	errCount=0
	for docIndex in testSet:
		wordVec=setOfWords2Vec(vocabList, docList[docIndex])
		if classifyNB(np.array(wordVec), p0V, p1V, pSpam)!=classList[docIndex]:
			errCount+=1
			print 'classify error ', docList[docIndex]
	print 'the error rate is: ', float(errCount)/len(testSet)

if __name__=='__main__':
	#testingNB()
	spamTest()
