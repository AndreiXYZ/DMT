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
from pyramid.arima import auto_arima


rootdir = 'datasets-per-pacient'
mse_all = []
pacient_indexes = []

train_data_all = []
forecasts_all = []
ground_truth_all = []

for findex,filename in enumerate(os.listdir(rootdir)):
	skip = 0
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

	# Keep last 10 days for testing our rolling prediction
	train, test = train_test_split(df, train_size=df.shape[0]-10 ,shuffle=False)

	# Create training and testing sets as np arrays
	x_train, y_train = train[exog_labels], train[endog_labels]
	x_train = np.asarray(x_train)
	y_train = np.asarray(y_train)

	x_test, y_test = test[exog_labels], test[endog_labels]
	x_test = np.asarray(x_test)
	y_test = np.asarray(y_test)

	history = x_train.copy()
	labels = y_train.copy()
	forecast = []

	train_data_all.append(y_train)
	for t in range(len(x_test)):

		try:
			model = auto_arima(y=y_train, exogenous=x_train, trace=False, error_action='ignore',
									   suppress_warnings=True, approx=False, stationary=False,
									   enforce_stationarity=False)
			model.fit(labels)
		except Exception as e:
			print(e)
			skip = 1
			pacient_indexes.append(findex)
			break
		
		prediction = model.predict(1)[0]
		forecast.append(prediction)
		history = np.append(history, x_test[t][np.newaxis, :], axis=0)
		labels = np.append(labels, y_test[t])
		print('Predicted: {} Expected: {}'.format(prediction, y_test[t]))

	if not skip:
		mse = mean_squared_error(forecast, y_test)
		print('Pacient {} MSE rolling prediction: {}'.format(findex,mse))
		mse_all.append(mse)
		forecasts_all.append(forecast)
		ground_truth_all.append(y_test)

print(len(mse_all))
print('Average MSE across all pacients: ', sum(mse_all)/len(mse_all))
print('Patients for which arima could not train: ', pacient_indexes)
with open('mse_all_arima.pkl', 'wb') as f:
	pickle.dump(mse_all, f)

with open('forecasts_vs_truth_arima.pkl', 'wb') as f:
	pickle.dump([forecasts_all, ground_truth_all, train_data_all, pacient_indexes], f)

print('Pacients for which arima could not train :', pacient_indexes)