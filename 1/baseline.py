import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

from pandas._libs.index import datetime, timedelta
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import numpy as np


def get_MSE(dataset):
    df = pickle.load(open(dataset, "rb"))
    predictions = []
    days = {}
    for time, mood in df[['time', 'mood']].values:
        date = time.date()
        days[date] = mood
        date += timedelta(days=1)
        predictions.append((date, mood))
    error = 0.0
    times = 0
    errors = []
    for prediction in predictions:
        date, mood = prediction
        if date in days.keys():
            err = (days[date] - mood) ** 2
            errors.append(err)
            error += err
            times += 1
    error /= times
    return error


errors = []
for filename in os.listdir("datasets-per-pacient"):
    error = get_MSE("datasets-per-pacient/" + filename)
    errors.append((error, filename))

with open('mse.pkl', 'wb') as f:
    pickle.dump(errors, f)

with open('mse.pkl', 'rb') as f:
    e = pickle.load(f)

for i, error in enumerate(errors):
    if i > 0:
        id = int(error[1].split('.')[1])
    else:
        id = 0
    errors[i] = (error[0], id)

y = []
x = []

mean = 0.0
maxi = 0.0
mini = 1000.0
for i,error in enumerate(errors):
    x.append(error[1])
    y.append(error[0])
    if i >-1:
        mean += y[-1]
        if y[-1] > maxi:
            maxi = y[-1]
        if y[-1] < mini:
            mini = y[-1]

mean /= len(errors)

variance = 0.0
for i,error in enumerate(errors):
    if i>-1:
        variance+= (error[1] -mean)**2

variance/= len(errors)
std_dev = np.sqrt(variance)
print('Avreage:' + str(mean))
print('Variance:' + str(variance))
print('Std dev:' + str(std_dev))
print('Max:' + str(maxi))
print('Min:' + str(mini))
plt.plot(x, y, label='MSE')
plt.grid()
plt.legend()
plt.title('MSE over all datasets')
plt.show()
