import pandas as pd
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import numpy as np
import os
import matplotlib.pyplot as plt

mseAllPatients=0
runsPerPatient=1
#m=[]
for file in os.listdir("datasets-per-pacient"):
	if 'AS' in file:
		df = pickle.load( open("datasets-per-pacient/"+file, "rb" ) )
		preprocessed=df.drop(['time','mood'],axis=1).rolling(6).mean().join(df['mood']).dropna()
		cor=preprocessed.corr()
		cor=cor['mood']
		drop=[]
		for idx,c in enumerate(cor):
			if np.abs(c)<0.05:
				drop.append(idx)
		print(len(drop))
		preprocessed=preprocessed.drop(preprocessed.columns[drop],axis=1)
		X=preprocessed.drop(['mood'],axis=1)
		y=preprocessed['mood']
		X=X.to_numpy(copy=True)
		y=y.to_numpy(copy=True)
		daystopred=1
		xtrain=X[0:-daystopred]
		xtest=X[-daystopred:]
		ytrain=y[0:-daystopred]
		ytest=y[-daystopred:]
		#totalmse=0
		for i in range(runsPerPatient):
			#xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.20)
			classifier=SVR(kernel='rbf',degree=2,gamma='auto')
			classifier.fit(xtrain,ytrain)
			pred=classifier.predict(xtest)
			actual=ytest
			mse=np.square(np.subtract(actual,pred)).mean()
			mseAllPatients+=mse
			plt.plot(np.append(ytrain,pred),'r')
			plt.plot(y,'b')
			plt.show()
			#totalmse+=mse
		#m.append(totalmse/runsPerPatient)
		#print(totalmse/runsPerPatient,file)
print(mseAllPatients/(27*runsPerPatient))
#plt.hist(m)
#plt.show()
