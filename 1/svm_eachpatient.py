import pandas as pd
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import numpy as np
import os
import matplotlib.pyplot as plt

mseAllPatients=0
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
		daystopred=10
		preds=[]
		for i in reversed(range(1,daystopred+1)):
			xtrain=X[0:-i]
			xtest=X[-i:]
			ytrain=y[0:-i]
			ytest=y[-i:]
			classifier=SVR(kernel='rbf',degree=2,gamma='auto')
			classifier.fit(xtrain,ytrain)
			#print(xtest)
			#print(xtest[0])
			#print(classifier.predict(xtest[0]))
			preds.append(classifier.predict([xtest[0]]))
			#actual=ytest[0]
		mse=np.square(np.subtract(y[-daystopred:],preds)).mean()
		mseAllPatients+=mse
		#print(y[-daystopred:])
		#print(preds)
		plt.plot(np.append(y[0:-daystopred],preds),'r')
		plt.plot(y,'b')
		plt.show()
			#totalmse+=mse
		#m.append(totalmse/runsPerPatient)
		#print(totalmse/runsPerPatient,file)
print(mseAllPatients/27)
#plt.hist(m)
#plt.show()
