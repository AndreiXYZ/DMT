import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dataset_mood_smartphone.csv')
df.rename(columns={'Unnamed: 0' : 'row'}, inplace=True)
# See how many entry points of each time we have
print(df.groupby('variable').count()['row'])