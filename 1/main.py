import pandas as pd
import seaborn as sns
import datetime
import pickle
import sys
import matplotlib.pyplot as plt

def convertToDatetime(dateString):
	dateString = dateString.split(' ')[0]
	return pd.to_datetime(dateString, format='%Y-%m-%d')

df = pd.read_csv('dataset_mood_smartphone.csv')
df.rename(columns={'Unnamed: 0' : 'row'}, inplace=True)

# Select just one patient for now
ids = df['id'].unique()


for id in ids:
	df = df[ df['id'] == id]
	# Store the means for mood, arousal and valence
	means = {}
	means['mood'] = df[ df['variable']=='mood'].mean().iloc[1]
	means['arousal'] = df[ df['variable']=='circumplex.arousal'].mean().iloc[1]
	means['valence'] = df[ df['variable']=='circumplex.valence'].mean().iloc[1]

	# Convert strings to datetime
	df['time'] = df['time'].apply(convertToDatetime)


	sms_calls = df[ (df['variable']=='sms') | (df['variable']=='call') ].groupby(['variable', 'time'], as_index=False).sum()
	meaned_values = df[ (df['variable']!='sms') & (df['variable'] != 'call') ].groupby(['variable', 'time'], as_index=False).mean()


	# print(df.groupby(['variable', 'time'], as_index=False).mean())
	fulldf = pd.concat([sms_calls, meaned_values], keys='time').reset_index().drop(['level_0', 'level_1', 'row'], axis=1)


	dates = sorted(fulldf['time'].unique())
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
				if value=='mood':
					if tempdict['mood']:
						tempdict[value].append(tempdict['mood'][-1])
					else:
						tempdict[value].append(means['mood'])
				elif value=='circumplex.arousal':
					if tempdict['circumplex.arousal']:
						tempdict[value].append(means['arousal'])
					else:
						tempdict[value].append(means['arousal'])
				elif value=='circumplex.valence':
					if tempdict['circumplex.valence']:
						tempdict[value].append(means['valence'])
					else:
						tempdict[value].append(means['valence'])
				else:
					tempdict[value].append(0)

	features_df = pd.DataFrame(tempdict)

	print(features_df.head)
	# corr_matrix = features_df.corr()
	# sns.heatmap(corr_matrix, 
	# 			xticklabels=corr_matrix.columns.values,
	# 			yticklabels=corr_matrix.columns.values)
	# plt.show()
	# print(corr_matrix)

	with open('processed_df_' + id + '.pkl', 'wb') as f:
		pickle.dump(features_df, f)