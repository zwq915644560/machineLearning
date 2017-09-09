#!/usr/bin/python
#-*- coding:utf-8 -*-

"my kNN"

__author__="zwq"

import numpy
import operator
import matplotlib
#two ways of creating data
def createDataSet():
	group=numpy.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels=['A','A','B','B']
	return group,labels

def file2Matrix(filename):
	fr=open(filename)
	dataOLines=fr.readlines()
	numOLines=len(dataOLines)
	featuresMat=numpy.zeros((numOLines,3))
	labelsVec=[]
	for i in range(numOLines):
		line=dataOLines[i].strip()
		listFromLine=line.split('\t')
		featuresMat[i,:]=[float(x) for x in listFromLine[:3]]#convert string to float explictly 
		labelsVec.append(int(listFromLine[-1]))
	return featuresMat,labelsVec

#analyze data
#...

#prepare data
def autoNorm(dataSet):
	minVals=dataSet.min(0)
	maxVals=dataSet.max(0)
	ranges=maxVals-minVals
	m=dataSet.shape[0]
	normDataSet=dataSet-numpy.tile(minVals,(m,1))
	normDataSet=normDataSet/numpy.tile(ranges, (m,1))
	return normDataSet,ranges,minVals

#function of classifing
def classify0(inX,dataSet,labels,k):
	dataSize=dataSet.shape[0]
	diffMat=dataSet-numpy.tile(inX,(dataSize,1))
	sqDiffMat=diffMat**2
	sqDistances=sqDiffMat.sum(axis=1)
	distances=sqDistances**0.5
	sortedDisIndices=distances.argsort()
	labelCount={}
	for i in range(k):
		label=labels[sortedDisIndices[i]]
		labelCount[label]=labelCount.get(label,0)+1
	sortedlabel=sorted(labelCount.iteritems(),key=operator.itemgetter(1),reverse=True)
	return sortedlabel[0][0]
	#return sortedlabel

#test the model
#...

#complete system
def classifyPerson():
	result=["in large does","in small does","not at all"]
	percentTats=float(raw_input("percentage of time spent in playing video games:"))
	ffMiles=float(raw_input("frequent flier miles earned per year:"))
	iceCream=float(raw_input("liters of ice cream consumed per year:"))
	dataSet,labels=file2Matrix("1.txt")
	normDataSet,ranges,minVals=autoNorm(dataSet)
	normInputData=([percentTats,ffMiles,iceCream]-minVals)/ranges
	classifierResult=classify0(normInputData, normDataSet, labels, 3)
	print "you will probably like this person: %s" %(result[classifierResult-1])

#test procedure
if __name__=="__main__":
	classifyPerson()
	