#!/usr/bin/python
# -*- coding:utf-8 -*-

'polynomial regression.'

__author__='zwq'

import numpy as np 
import matplotlib.pyplot as plt 
import pickle
import LinRegre as lr
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

def loadDataSet():
	fr1=open('./datas')
	X=pickle.load(fr1)
	fr2=open('./targets')
	y=pickle.load(fr2)
	return np.array([e[1] for e in X]).reshape(-1,1), np.array(y)


def polyFeatures(X_train, d):
	poly=PolynomialFeatures(degree=d, interaction_only=False)
	return poly.fit_transform(X_train)

def plotPoly(theta, X_train, y_train):
	x=[e[1] for e in X_train]
	y=y_train
	fig=plt.figure()
	axe=fig.add_subplot(111)
	axe.plot(x, y, 'bo')
	
	#x1=np.linspace(0,1,1000)
	x1=np.array(x)
	y1=theta[0]+theta[1]*x1+theta[2]*x1**2
	#y1=theta[0]+theta[1]+theta[2]*x1+theta[3]+theta[4]*x1+theta[5]*x1**2+theta[6]+theta[7]*x1+theta[8]*x1**2+theta[9]*x1**3
	axe.plot(x1, y1, 'r-')
	plt.show()


X_train, y_train=loadDataSet()
#X=polyFeatures(X_train, 2)
#print X

#ss=StandardScaler()
#X=ss.fit_transform(X)

#theta, thetaCache=lr.get_theta(X, y_train, alpha=0.1, iter=250)
#print thetaCache, theta
models=[
	Pipeline([
		('Poly', PolynomialFeatures()),
		('Linear', LinearRegression())
	])
]
model=models[0]
for d in range(1,20):
	model.set_params(Poly__degree=d)
	model.fit(X_train, y_train)
	
	lin=model.get_params('Linear')['Linear']
	print '%d阶线性回归：准确率：%.3f' % (d, model.score(X_train, y_train))
	print '系数：\n', lin.coef_
	#if d==2: plotPoly(lin.coef_, X_train, y_train)
#lr.plotJWithIterion(thetaCache, 250, X, y_train)
#plotPoly(theta, X_train, y_train)
