#!/usr/bin/env python
# coding: utf-8

# # Linear Regression Using StatsModels and Linear_Model

# Import Libraries for the Supervised Learning Algorithm 


import pandas as pd
import numpy as np
import ppscore as pps
import re
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
import requests
import json
from pandas.io.json import json_normalize
import xlsxwriter
import sys
import pprint
from collections import namedtuple
from urllib.request import urlopen
from matplotlib.pylab import rcParams
import warnings
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets, linear_model
import statsmodels.api as sm


# Convert JSON API to Dataframe Structure 
    #''' JSON TO Pandas DataFrame '''
    ## Fetch json data 
    #res = requests.get("https://api-json.com",headers={"Authorization":"Basic dfuTWOBk"})
    #request=res.text
    #data=json.loads(request)
    ###################################################


bjson= pd.read_json("C:/Desktop/b_data.json")
l = []
for k, v in sorted(bjson.items()):
    l.extend(v)
#print(l)
df=pd.DataFrame(l)


# Read Dataframe which was converted from JSON API

df.head()

df.columns

df.shape

df.describe()


# Using Seaborn library we can check null values through visual interface


sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# Through Numeric as well we can get complete details of NULL Values from data

df.isnull().sum()

#df['last_name']=df['last_name'].astype('category') 

df.dtypes


# Using CountPlot from seaborn library we can draw and check unique & nunique categories for a particular variable


plt.figure(figsize=(12,6))
sns.countplot(x='last_name', data=df)
plt.xticks(rotation='vertical')
plt.show()

df.dtypes
# While dealing with dataset , we may seen str/categorical/object datatypes that all need to converted to numeric way for building a models


le = LabelEncoder()
df["last_name"] =le.fit_transform(df["last_name"].astype('str'))


# With correalation / covariance we can get importance of all variables from dataframe



df.corr()


# # Using StatsModel Implemented Linear regression 

Sm_Model= sm.OLS.from_formula("surgeries_this_month~surgeries_last_month",data=df)
result_ln=Sm_Model.fit()
print("Linear_Regression",result_ln.summary())


# # Using StatsModel Implemented Multiple Linear regression 


Sm_Model= sm.OLS.from_formula("surgeries_this_month~age_in_yrs+hospital_id+last_name+service_id+surgeries_last_month",data=df)
result_ln=Sm_Model.fit()
print("Multiple_Linear_Regression",result_ln.summary())


# # Using Sklearn Impemented Linear_Model 

y=df['surgeries_this_month']
df=df.drop(['surgeries_this_month'],axis=1)
df


# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# fit a model
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
predictions


# # With Simple Scatter plots we can draw predictions

plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")


print("Score:",model.score(X_test, y_test))

