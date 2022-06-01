import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('framingham.csv')
df.corr()
df.drop(columns=['education'],inplace=True)
df.replace(np.nan,0,inplace=True)
df.drop(columns=['cigsPerDay','diaBP'],inplace=True)

df=df.loc[~(df['totChol']>300)]
df=df[~(df['sysBP']>200)]
df=df[~(df['BMI']>40)]
df=df[~(df['heartRate']>110)]
df=df[~(df['glucose']>150)]

df = df.reset_index()
df.drop(columns=['index'],inplace=True)

#x_one = df[:3000]
#x_one = x_one.loc[x_one['TenYearCHD'] == 1]

y = df['TenYearCHD']
x = df.drop(columns=['TenYearCHD'])

## train test split
lim = 3000
# Normal data
x_train = x[:lim]
y_train = y[:lim]

x_test = x[lim:]
y_test = y[lim:]


##Pre processing
from sklearn.preprocessing import StandardScaler
s=StandardScaler()
x_train=s.fit_transform(x_train)
x_test=s.fit_transform(x_test)

## Liner Regression
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
l=LogisticRegression()
l.fit(x_train,y_train)

## dumpling the model in pickle file
pickle.dump(l, open('model_ll.pkl','wb'))

