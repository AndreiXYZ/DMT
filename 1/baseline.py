import pandas as pd
import pickle

from pandas._libs.index import datetime, timedelta
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import numpy as np

df = pickle.load(open("datasets-per-pacient/processed_df_AS14.01.pkl", "rb"))
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
        print('Actual: ' + str(days[date]))
        print('Prediction: ' + str(mood))
        err = abs(days[date] - mood)
        errors.append(err)
        error += err
        times += 1
error /= times
print("Error:" + str(error))
