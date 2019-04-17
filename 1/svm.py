import pandas as pd
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import numpy as np

df = pickle.load( open( "datasets-per-pacient/processed_df_all_pacients.pkl", "rb" ) )
print(df)
preprocessed=df.drop(['time','mood'],axis=1).rolling(6).mean().join(df['mood']).dropna()
print(preprocessed)
X=preprocessed.drop(['mood'],axis=1)
y=preprocessed['mood']
xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.20)
classifier=SVR(kernel='rbf')


classifier.fit(xtrain,ytrain)
pred=classifier.predict(xtest)
print('pred',pred)
actual=ytest.values
print('actual',actual)
print('mse',np.square(np.subtract(actual,pred)).mean())
