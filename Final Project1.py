import pandas as pd
import numpy as np
import copy

# -- import data --
df = pd.read_csv("subscribers.csv")

# -- keep the columns we need --
column_list = ['package_type','preferred_genre','intended_use','weekly_consumption_hour','age','male_TF']
data1 = df[column_list]

# -- remove missing values --
data1 = data1.dropna()

# -- process extreme data --
data1.drop(data1[data1.age > 80].index, inplace=True)
data1.drop(data1[data1.age < 15].index, inplace=True)

# -- covert categorical variables to dummies --
data2 = pd.get_dummies(data = data1, columns=['package_type', 'preferred_genre', 'intended_use','male_TF'])
print(data2.columns)
print(data2.shape)

# -- export --
data_cleaned_dummified = data2.copy()
data_cleaned_dummified.to_csv('subscribers_cleaned_dummified.csv', index=False)

