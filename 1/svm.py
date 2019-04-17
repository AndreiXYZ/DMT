from sklearn import svm
import pandas as pd

dp = pd.read_pickle('./datasets-per-pacient/processed_df_AS14.01.pkl')
# preporcesing data, averaging features over 5 days
timestamps = dp['time']
days = []
for timestamp in timestamps:
    days.append(timestamp.date())

dp['time'] = days
for day in days:
    five_nearest_days = []
    days_distance = dp['time'] - day
    dates = dp.loc[(dp['time'] - day )]
    #print(dates)
    break
