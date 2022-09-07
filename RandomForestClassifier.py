#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split




data= pd.read_csv("G:\cardio_train.csv",sep=';')


data['cardio'].value_counts()
data.describe()
data.drop(['id'],axis=1)
x= data.iloc[:,:-1]

y=data.iloc[:,-1]
 
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=.3)



 
print('Random Forest Accuracy: ')
rfc=RandomForestClassifier();
rfc.fit(xtrain,ytrain)
rfc.score(xtest,ytest)

sc=(rfc.score(xtest,ytest))*100
print(sc)

