import pandas as pd
import matplotlib.pyplot as plt
import datetime

def convertToDatetime(dateString):
	dateString = dateString.split(' ')[0]
	return pd.to_datetime(dateString, format='%Y-%m-%d')

df = pd.read_csv('dataset_mood_smartphone.csv')
df.rename(columns={'Unnamed: 0' : 'row'}, inplace=True)

# Select just one patient for now
df = df[ df['id'] == 'AS14.01']

# Convert strings to datetime
df['time'] = df['time'].apply(convertToDatetime)

sms_calls = df[ (df['variable']=='sms') | (df['variable']=='call') ].groupby(['variable', 'time'], as_index=False).sum()
meaned_values = df[ (df['variable']!='sms') & (df['variable'] != 'call') ].groupby(['variable', 'time'], as_index=False).mean()

# print(df.groupby(['variable', 'time'], as_index=False).mean())
fulldf = pd.concat([sms_calls, meaned_values], keys='time')
print(sms_calls.shape)
print(meaned_values.shape)
print(fulldf)