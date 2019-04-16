import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pickle

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
fulldf = pd.concat([sms_calls, meaned_values], keys='time').reset_index().drop(['level_0', 'level_1', 'row'], axis=1)

dates = fulldf['time'].unique()
vals = fulldf['variable'].unique()

tempdict = {key: [] for key in vals}
tempdict['time'] = []

for date in dates:
	tempdict['time'].append(date)
	for value in vals:
		selection = fulldf[ (fulldf['time'] == date) & (fulldf['variable'] == value) ]
		try:
			tempdict[value].append(selection.iloc[0]['value'])
		except Exception as e:
			# TODO: Process mood, arousal and valence
			tempdict[value].append(0)

features_df = pd.DataFrame(tempdict)



with open('processed_df.pkl', 'wb') as f:
	pickle.dump(features_df, f)