import pickle
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def undifference_result(res, original_data):
	res += original_data[0]
	# res = res.cumsum()
	return res

# Get dataset
with open('datasets-per-pacient/processed_df_all_pacients.pkl', 'rb') as f:
	df = pickle.load(f)
# Drop timestamps since they are no longer needed
df = df.drop('time', axis=1)

# Plot autocorrelation

# plot_acf(df['mood'])
# plt.show()

# Split into training and validation
train, test = train_test_split(df, train_size=0.8 ,shuffle=False)

# Prepare exogenous and endogenous labels
exog_labels = df.columns.tolist()
exog_labels.remove('mood')
endog_labels = 'mood'

# Create training and testing sets as np arrays
x_train, y_train = train[exog_labels], train[endog_labels]
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

x_test, y_test = test[exog_labels], test[endog_labels]
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
# Define model
model = ARIMA(exog=x_train, endog=y_train, order=(5,1,0))
model_fit = model.fit(max_iter=500, disp=10)

# In sample prediction to get train error
# starts from 1 since 1st element is differenced away
yhat_train = model_fit.predict(start=1)
yhat_train = undifference_result(yhat_train, y_train)

print('Train MSE :', mean_squared_error(yhat_train, y_train[1:]))

# yhat_test = model_fit.predict(start=len(x_train), end=len(x_train)+len(x_test)-1,
# 												exog=x_test, dynamic=False)

yhat_test = model_fit.predict(start=len(x_train), end=len(x_train)+len(x_test)-1, 
							  exog=x_test)

yhat_test = undifference_result(yhat_test, y_test)
# yhat_test = undifference_result(yhat_test, y_test)
print('Test MSE :', mean_squared_error(yhat_test[:-3], y_test[:-3]))

plt.subplot(2, 1, 1)
plt.plot(y_train, label='y train')
plt.plot(yhat_train, label='yhat train')
plt.grid()
plt.legend()
plt.title('In-sample predictions')

plt.subplot(2, 1, 2)
plt.plot(y_test[:-3], label='y test')
plt.plot(yhat_test[:-3], label='yhat test')
plt.grid()
plt.legend()
plt.title('Out of sample forecast')
plt.show()