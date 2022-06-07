#importing all necessary files
import pandas as pd
import numpy as np
import pickle

# imporing the dataset
df = pd.read_csv('framingham.csv')

# data corelation
df.corr()

#feature selection by droping unwanted features
df.drop(columns=['education'],inplace=True)
df.drop(columns=['cigsPerDay','diaBP'],inplace=True)

# Replaceing all null values with mean of the features
df.replace(np.nan,df.mean(),inplace=True)

# Removing the instances which do not contribute to the Accuracy
df=df.loc[~(df['totChol']>300)]
df=df[~(df['sysBP']>200)]
df=df[~(df['BMI']>40)]
df=df[~(df['heartRate']>110)]
df=df[~(df['glucose']>150)]

# Resitting the index
df = df.reset_index()
df.drop(columns=['index'],inplace=True)

# splitting the target feature from the dataset
y = df['TenYearCHD']
x = df.drop(columns=['TenYearCHD'])

## splitting training and testing data
lim = 3041

# Training data
x_train = x[:lim]
y_train = y[:lim]

#testing data
x_test = x[lim:]
y_test = y[lim:]


##Pre-processing
from sklearn.preprocessing import StandardScaler
s=StandardScaler()
x_train=s.fit_transform(x_train)
x_test=s.fit_transform(x_test)

## Logistic Regression Algorithm
from sklearn.linear_model import LogisticRegression
l = LogisticRegression(solver='liblinear',max_iter = 1000, random_state = 31)
l.fit(x_train,y_train)

# KNN Algorithm
from sklearn.neighbors import KNeighborsClassifier
k=KNeighborsClassifier(n_neighbors=12)
k.fit(x_train,y_train)

# Decision Tree ALgorithm
from sklearn.tree import DecisionTreeClassifier
d=DecisionTreeClassifier()
d.fit(x_train,y_train)

# Support Vector Machine Algorithm
from sklearn.svm import SVC
s=SVC(C=10,kernel="linear",gamma="auto")
s.fit(x_train,y_train)

# Random Forest Algorithm
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=168)
rf.fit(x_train,y_train)


## dumpling the model in pickle file
pickle.dump(l, open('model_lr.pkl','wb'))
pickle.dump(k, open('model_knn.pkl','wb'))
pickle.dump(d, open('model_dt.pkl','wb'))
pickle.dump(s, open('model_svm.pkl','wb'))
pickle.dump(rf, open('model_rf.pkl','wb'))

