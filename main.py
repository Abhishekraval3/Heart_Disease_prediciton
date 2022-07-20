#importing all necessary files
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score

# imporing the dataset
df = pd.read_csv('framingham.csv')

# data corelation
df.corr()

#feature selection by droping unwanted features
df.drop(columns=['education'],inplace=True)
df.drop(columns=['cigsPerDay'],inplace=True)

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

# Testing data
x_test = x[lim:]
y_test = y[lim:]


## Logistic Regression Algorithm
from sklearn.linear_model import LogisticRegression
l = LogisticRegression(solver='liblinear',max_iter = 1000, random_state = 0)
l.fit(x_train,y_train)
lr_train_pred = l.predict(x_train)
lr_test_pred = l.predict(x_test)
lr_train_acc = round(accuracy_score(lr_train_pred,y_train)*100,2)
lr_test_acc = round(accuracy_score(lr_test_pred,y_test)*100,2)


# KNN Algorithm
from sklearn.neighbors import KNeighborsClassifier
k=KNeighborsClassifier(n_neighbors=13)
k.fit(x_train,y_train)
knn_train_pred = k.predict(x_train)
knn_test_pred = k.predict(x_test)
knn_train_acc = round(accuracy_score(knn_train_pred,y_train)*100,2)
knn_test_acc = round(accuracy_score(knn_test_pred,y_test)*100,2)


# Decision Tree ALgorithm
from sklearn.tree import DecisionTreeClassifier
d=DecisionTreeClassifier()
d.fit(x_train,y_train)
dt_train_pred = d.predict(x_train)
dt_test_pred = d.predict(x_test)
dt_train_acc = round(accuracy_score(dt_train_pred,y_train)*100,2)
dt_test_acc = round(accuracy_score(dt_test_pred,y_test)*100,2)


# Support Vector Machine Algorithm
from sklearn.svm import SVC
s=SVC(C=10,kernel="linear",gamma="auto")
s.fit(x_train,y_train)
svm_train_pred = s.predict(x_train)
svm_test_pred = s.predict(x_test)
svm_train_acc = round(accuracy_score(svm_train_pred,y_train)*100,2)
svm_test_acc = round(accuracy_score(svm_test_pred,y_test)*100,2)


# Random Forest Algorithm
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=160)
rfr = RandomForestClassifier(random_state=62)
rf.fit(x_train,y_train)
rf_train_pred = rf.predict(x_train)
rf_test_pred = rf.predict(x_test)
rf_train_acc = round(accuracy_score(rf_train_pred,y_train)*100,2)
rf_test_acc = round(accuracy_score(rf_test_pred,y_test)*100,2)

#Ensamble Model using Stacking
from sklearn.ensemble import StackingClassifier
estimators = {('knn',k),
             ('svm',s),
              ('dt',d),
              ('lr',l),
              ('rf',rfr)}

stack_model = StackingClassifier(
estimators = estimators, final_estimator = LogisticRegression(random_state=0))
stack_model.fit(x_train,y_train)
stack_train_pred = stack_model.predict(x_train)
stack_test_pred = stack_model.predict(x_test)
stack_train_acc = round(accuracy_score(stack_train_pred,y_train)*100,2)
stack_test_acc = round(accuracy_score(stack_test_pred,y_test)*100,2)

#printing Accuracy
print('LR_training_acc',lr_train_acc)
print('LR_test_acc',lr_test_acc)

print('KNN_training_acc',knn_train_acc)
print('KNN_test_acc',knn_test_acc)

print('DT_training_acc',dt_train_acc)
print('DT_test_acc',dt_test_acc)

print('SVM_training_acc',svm_train_acc)
print('SVM_test_acc',svm_test_acc)

print('RF_training_acc',rf_train_acc)
print('RF_test_acc',rf_test_acc)

print('STACK_training_acc',stack_train_acc)
print('STACK_test_acc',stack_test_acc)

## dumpling the model in pickle file
pickle.dump(l, open('model_lr.pkl','wb'))
pickle.dump(k, open('model_knn.pkl','wb'))
pickle.dump(d, open('model_dt.pkl','wb'))
pickle.dump(s, open('model_svm.pkl','wb'))
pickle.dump(rf, open('model_rf.pkl','wb'))
pickle.dump(stack_model,open('model_stack.pkl','wb'))
