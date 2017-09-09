#!/usr/bin/python
# -*- coding:utf-8 -*-

'linear regression.'

__author__='zwq'

import numpy as np 
import matplotlib.pyplot as plt 
import pickle

def loadDataSet(path):
	import re
	X=[];y=[]
	fr=open(path)
	for line in fr.readlines():
		li=re.split(r'[\s\:]',line.strip())
		del(li[0])
		del(li[2])
		tmp=[]
		for num in li[:-1]:
			tmp.append(float(num))
		X.append(tmp)
		y.append(float(li[-1]))

	fw1=open('datas', 'wb')
	fw2=open('targets', 'wb')
	pickle.dump(X, fw1)
	pickle.dump(y, fw2)
	
	return np.array(X), np.array(y)

#loadDataSet('2_dim_datas.txt')

# hypothesis/predict function
def h(theta, input):
	return input.dot(theta)
# cost function
def j(theta, X_train, y_train):
	m=X_train.shape[0]
	err=h(theta, X_train)-y_train.reshape(-1,1)
	return np.sum(err**2)/(2*m)
# learning algorithm
def get_theta(X_train, y_train, alpha=0.1, iter=250):
	#X_train=np.array(X_train)
	#y_train=np.array(y_train)
	m,n=X_train.shape
	theta=np.zeros((n, 1))
	thetaCache=[]
	thetaCache.append(theta)
	m,n=X_train.shape
	for i in range(iter):
		err=h(theta, X_train)-y_train.reshape(-1,1)
		grad=(1.0/m)*np.sum(np.tile(err, (1,n))*X_train,
			axis=0)
		tmp=theta-alpha*grad.reshape(-1, 1)
		thetaCache.append(tmp)
		theta=tmp
	return theta, thetaCache


def plotJWithIterion(thetaCache, iter, X_train, y_train):
	x=range(iter+1)
	y=[]
	for theta in thetaCache:
		y.append(j(theta, X_train, y_train))
	fig=plt.figure()
	axe=fig.add_subplot(111)
	axe.plot(x, y, 'r-')
	plt.xlabel('iter')
	plt.ylabel('cost')
	plt.title('cost function with the iteration')
	plt.show()

def plot(theta, X_train, y_train):
	x=[e[1] for e in X_train]
	y=y_train
	fig=plt.figure()
	axe=fig.add_subplot(111)
	axe.plot(x, y, 'bo')
	
	x1=np.linspace(0,1,1000)
	y1=theta[1]*x1+theta[0]
	axe.plot(x1, y1, 'r-')
	plt.show()

if __name__=='__main__':
	X_train, y_train=loadDataSet('2_dim_datas.txt')
	#print X_train, y_train
	theta, thetaCache=get_theta(X_train, y_train)
	#print thetaCache, theta
	plotJWithIterion(thetaCache, 250, X_train, y_train)
	plot(theta, X_train, y_train)
