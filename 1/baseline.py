import pandas as pd
import pickle
import os

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
    return np.sqrt(error)


errors = []
for filename in os.listdir("datasets-per-pacient"):
    error = get_MSE("datasets-per-pacient/" + filename)
    errors.append((error, filename))

for error in errors:
    print(error)