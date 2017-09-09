#!/usr/bin/python
# -*- coding:utf-8 -*-

'logistic regression.'

__author__='zwq'

import numpy as np 
import matplotlib.pyplot as plt 
from copy import deepcopy

# obtain data
def loadDataSet(path):
	dataMat=[]; labelMat=[]
	fr=open(path)
	for line in fr.readlines():
		lineArr=line.strip().split()
		dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
		labelMat.append(int(lineArr[2]))
	return dataMat, labelMat

def loadDataSet1(path):
	fr=open(path)
	feaSet=[]; Labels=[]
	for line in fr.readlines():
		lineArr=line.strip().split()
		tmp=[]
		for x in lineArr[:-1]:
			tmp.append(float(x))
		feaSet.append(tmp)
		Labels.append(float(lineArr[-1]))
	return feaSet, Labels

# 
def sigmoid(x):
	return 1.0/(1+np.exp(-x))

# train the algorithm
def gradAscent(dataMat, labelMat):
	www=[]
	dataMatrix=np.array(dataMat)
	labelVec=np.array(labelMat).reshape(-1,1)
	n=dataMatrix.shape[1]
	weights=np.ones((n, 1))
	alpha=0.001
	maxCycles=500
	for i in range(maxCycles):
		h=sigmoid(np.dot(dataMatrix, weights))
		errVec=labelVec-h
		weights+=alpha*dataMatrix.T.dot(errVec)
		www.append(deepcopy(weights))
	return weights, www

def stocGradAscent0(dataMat, labelMat):
	www=[]
	dataMatrix=np.array(dataMat)
	labelVec=np.array(labelMat).reshape(-1,1)
	n=dataMatrix.shape[1]
	weights=np.ones((n, 1))
	alpha=0.01
	iter=150
	for _ in range(iter):
		for i in range(len(dataMat)):
			h=sigmoid(dataMatrix[i].dot(weights))
			err=labelVec[i]-h
			weights+=alpha*dataMatrix[i].reshape(-1,1)*err
			www.append(deepcopy(weights))	
	return weights, www

def stocGradAscent1(dataMat, labelMat, numIter=150):
	www=[]
	dataMatrix=np.array(dataMat)
	labelVec=np.array(labelMat).reshape(-1,1)
	m, n=dataMatrix.shape
	weights=np.ones((n, 1))
	
	for j in range(numIter):
		dataIndex=range(m)
		for i in range(m):
			alpha=4/(1.0+i+j)+0.01
			randIndex=int(np.random.uniform(0, len(dataIndex)))
			h=sigmoid(dataMatrix[dataIndex[randIndex]].dot(weights))
			err=labelVec[dataIndex[randIndex]]-h
			weights+=alpha*dataMatrix[dataIndex[randIndex]].reshape(-1,1)*err
			del(dataIndex[randIndex])
			www.append(deepcopy(weights))	
	return weights, www

# analyze data
def plotBestFit(dataMat, labelMat, weights):
	x1=[];y1=[]
	x2=[];y2=[]
	for i in range(len(labelMat)):
		if labelMat[i]==1:
			x1.append(dataMat[i][1])
			y1.append(dataMat[i][2])
		elif labelMat[i]==0:
			x2.append(dataMat[i][1])
			y2.append(dataMat[i][2])
	x=np.arange(-4.0, 4.0, 1)
	y=-(weights[0]+weights[1]*x)/weights[2]
	fig=plt.figure()
	ax=fig.add_subplot(1,1,1)
	ax.scatter(x1, y1, c='r', s=30, marker='s')
	ax.scatter(x2, y2, c='g', s=30, marker='o')
	ax.plot(x, y)
	ax.set_xlabel('x1')
	ax.set_ylabel('y1')
	ax.set_title('data analyzing')
	plt.show()
	return fig

def plotWeight(www):
	w0=[]; w1=[]; w2=[]
	for ww in www:
		w0.extend(ww[0])
		w1.extend(ww[1])
		w2.extend(ww[2])
	x=range(1, len(w0)+1);
	
	fig=plt.figure()
	ax=fig.add_subplot(111)
	ax.plot(x, w0, c='r', label='w0')
	ax.plot(x, w1, c='g', label='w1')
	ax.plot(x, w2, c='b', label='w2')
	plt.legend(loc='best')
	plt.xlabel('iteration')
	plt.ylabel('weights')
	plt.show()
	return fig

# application
def classifyVector(inX, weights):
	inVec=np.array(inX)
	prob=sigmoid(inVec.dot(weights))
	return 1 if prob>0.5 else 0

def colicTest():
	trainingSet, trainingLabel=loadDataSet1('./samples/horseColicTraining.txt')
	weights, www=stocGradAscent1(trainingSet, trainingLabel, 500)

	testSet, testLabel=loadDataSet1('./samples/horseColicTest.txt')
	errCount=0.0
	for i in range(len(testSet)):
		out=classifyVector(testSet[i], weights)
		if out!=testLabel[i]:
			errCount+=1
	errRate=errCount/len(testLabel)
	print 'the error rate of this test is:%f' %errRate
	return errRate

def multiTest():
	numTests=10; errSum=0.0
	for _ in range(numTests):
		errSum+=colicTest()
	print 'after %d iterations, the average error rate is:%f' %(numTests, errSum/numTests)



if __name__=='__main__':
	'''
	dataMat, labelMat=loadDataSet('./samples/testSet.txt')
	
	weights, www=stocGradAscent1(dataMat, labelMat)
	print weights
	fig=plotWeight(www)
	fig.savefig('./figures/stocGradAscent1')
	figFit=plotBestFit(dataMat, labelMat, weights)
	figFit.savefig('./figures/stocGradAscent1Fit')
	'''
	multiTest()