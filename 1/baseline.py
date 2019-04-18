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
            err = (days[date] - mood)**2
            errors.append(err)
            error += err
            times += 1
    error /= times
    return error


errors = []
for filename in os.listdir("datasets-per-pacient"):
    error = get_MSE("datasets-per-pacient/" + filename)
    errors.append((error, filename))



for i,error in enumerate(errors):
    if i>0:
        id = int(error[1].split('.')[1])
    else:
        id = 0
    errors[i] = (error[0],id)



y = []
x = []

for error in errors:
    x.append(error[1])
    y.append(error[0])

plt.plot(x,y, label='MSE')
plt.grid()
plt.legend()
plt.title('MSE over all datasets')
plt.show()