import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sizes = []
moods = []
rootdir = 'datasets-per-pacient'
mood_df = pd.DataFrame()
for idx,fname in enumerate(os.listdir(rootdir)):
	if 'all' in fname:
		continue
	with open(rootdir + '/' + fname, 'rb') as f:
		df = pickle.load(f)
	sizes.append(df.shape[0])
	if idx==0:
		mood_df = df['mood']
	else:
		print(mood_df)
		mood_df = mood_df.append(df['mood'], ignore_index=True)

# plt.bar(np.arange(27), sizes)
# plt.title('Number of days recorded per patient')
# plt.grid()
# plt.ylabel('Days recorded')
# plt.xlabel('Patient')
# plt.show()




print(mood_df)
with open(rootdir + '/' + 'processed_df_all_pacients.pkl', 'rb') as f:
	df = pickle.load(f)

mood_df.hist(bins=20)
print(mood_df.describe())
plt.title('Average mood per day')
plt.show()