import pickle
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima


rootdir = 'datasets-per-pacient'
mse_all = []

for findex,filename in enumerate(os.listdir(rootdir)):
	if 'all' in filename:
		continue

	with open(rootdir + '/' + filename, 'rb') as f:
		df = pickle.load(f)

	# Drop time since we don't need it anymore
	df = df.drop('time', axis=1)

	# Drop useless features
	cor = df.corr()
	cor=cor['mood']
	drop=[]
	for idx,c in enumerate(cor):
			if np.abs(c)<0.05:
				drop.append(idx)
	df = df.drop(df.columns[drop],axis=1)
	print('Dropped {} cols'.format(len(drop)))

	# Prepare exogenous and endogenous labels
	exog_labels = df.columns.tolist()
	exog_labels.remove('mood')
	endog_labels = 'mood'
	train, test = train_test_split(df, train_size=0.8 ,shuffle=False)

	# Create training and testing sets as np arrays
	x_train, y_train = train[exog_labels], train[endog_labels]
	x_train = np.asarray(x_train)
	y_train = np.asarray(y_train)

	x_test, y_test = test[exog_labels], test[endog_labels]
	x_test = np.asarray(x_test)
	y_test = np.asarray(y_test)

	try:
		model = auto_arima(y=y_train, exogenous=x_train, trace=True, error_action='ignore',
							   suppress_warnings=True, approx=False, D=1, stationary=False,
							   max_order=12)
		model.fit(y_train)
	except Exception as e:
		print(e)

	forecast = model.predict(len(x_test))
	mse = mean_squared_error(forecast, y_test)
	print('Pacient {} MSE: {}'.format(findex,mse))
	mse_all.append(mse)

print(len(mse_all))
print('Average MSE across all pacients: ', sum(mse_all)/len(mse_all))