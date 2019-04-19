import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

with open('forecasts_vs_truth_arima.pkl', 'rb') as f:
	resultList = pickle.load(f)

forecasts_all = resultList[0] 
labels_all = resultList[1]
train_data_all = resultList[2]
pacient_indexes = resultList[3]

print(forecasts_all)
results = []
mses = []
for forecast, label in zip(forecasts_all, labels_all):
	mses.append(mean_squared_error(forecast, label))

print('Overall mse:', sum(mses)/len(mses))
# del train_data_all[19]
# del train_data_all[17]

# forecasts_all = np.array(forecasts_all)
# labels_all = np.array(labels_all)
# train_data_all = np.array(train_data_all)

# print(forecasts_all.shape)
# print(labels_all.shape)

# print(train_data_all.shape)
# print(forecasts_all[0])
# print(labels_all[0])
# print(train_data_all[0])
# for i in range(25):
# 	ground_truth = np.append(train_data_all[i],labels_all[i])
# 	plt.plot(ground_truth, label='ground truth')
# 	plt.plot(range(len(train_data_all[i]),len(train_data_all[i])+10), forecasts_all[i], 'red', label='forecast')
# 	plt.legend()
# 	plt.title('ARIMA forecast for one patient')
# 	plt.grid()
# 	plt.show()