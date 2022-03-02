import itertools
import numpy as np
import pandas as pd
import scipy
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Read Dataset
df = pd.read_csv('dataset.csv', sep=';')
print(df.shape)

# Drop columns and last strings with Nan
df.drop(["Unnamed: 0", "Unnamed: 0.1"], axis = 1, inplace = True)
N = df.shape[0]
df = df.iloc[: (N - 10), :]

# Change school_value
school_value = df[["school", "target"]].groupby('school').agg('mean')
df["school"] = df["school"].apply(lambda x: school_value.loc[x])

# One-Hot Encoding with Pandas
categorical_features = df.columns[df.dtypes == 'object']
df = pd.get_dummies(df, columns = categorical_features, drop_first = True)

def find_most_similar_strings(df, st):
    max = 0
    ages = []
    for st1 in df.index:
        sum = 0
        if (st1 != st):
            for col in df.columns:
                sum += int(df.loc[st, col] == df.loc[st1, col])
            if (sum > max):
                max = sum
                ages = []
                age = df.loc[st1, "n_student"]
                if not (np.isnan(age)):
                    ages.append(age)
            elif (sum == max):
                age = df.loc[st1, "n_student"]
                if not (np.isnan(age)):
                    ages.append(age)
    return np.median(np.array(ages))

# Fill Nan with age = age in similar strings
for st in df.index:
    if (np.isnan(df.loc[st, 'n_student'])):
        df.loc[st, 'n_student'] = find_most_similar_strings(df, st)

# Create new feature
for i in df.gender_Male.unique():
    for j in df.n_student.unique():
        l=df.loc[(df.gender_Male==i) & (df.n_student>=j)]
        woe=l.target.sum()/(l.target.count()-df.target.mean()-1)
        df.loc[(df.gender_Male==i) & (df.n_student>=j), 'woe_agegender']=woe

# Min-Max scale for age and woe
scaler = MinMaxScaler()
df['n_student'] = scaler.fit_transform(pd.DataFrame(df['n_student']))
df['woe_agegender'] = scaler.fit_transform(pd.DataFrame(df['woe_agegender']))

# Drop strings (around 10) with nans (similar strings with nan too)
df = df.fillna(df.mean())

# Create X and Y
X = df.drop(['posttest'], axis=1)
X = X.drop(['target'], axis=1)
y = df['posttest']

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234)

# Learn model
LR = LinearRegression()
LR.fit(X_train, y_train)
y_pred = LR.predict(X_train)

# Result
print(mean_absolute_error(y_test, LR.predict(X_test)))

