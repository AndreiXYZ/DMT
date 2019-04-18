import pandas as pd
import pickle

from pandas._libs.index import datetime, timedelta
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import numpy as np

IDS = ["01","02"]
df = pickle.load(open("./datasets-per-pacient/processed_df_all_pacients.pkl", "rb"))
print(df['time'])
for w in df[:]:
    print(w)
