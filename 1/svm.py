import pandas as pd
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pickle.load( open( "datasets-per-pacient/processed_df_AS14.01.pkl", "rb" ) )
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
#corr_matrix = preprocessed.corr()
#sns.heatmap(corr_matrix, 
#			xticklabels=corr_matrix.columns.values,
# 			yticklabels=corr_matrix.columns.values)
#plt.show()
#print(corr_matrix['mood'])