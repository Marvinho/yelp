# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 01:41:49 2018

@author: marvi
"""
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import numpy as np
from scipy import stats





path = "F:/Downloads/yelp_dataset/yelp_dataset/"
filename = "yelp_academic_dataset_business.json"




df = pd.read_json(path+filename, lines = True)

#print(len(df))
#print(df.head)  

#print(list(df))

df = df[["review_count", "stars", "is_open"]]
df = df[df["review_count"] < 2*df["review_count"].std()]

print(df.describe())

dff = df["is_open"].value_counts()


#sns.catplot(x="stars", y="review_count", hue="is_open", kind="swarm", data=df);


sns.pairplot(df)
#x = df.values #returns a numpy array
#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(x)
#df = pd.DataFrame(x_scaled)
#sns.pairplot(df)

#df.plot.scatter(x="review_count", y="is_open")
#df.plot.scatter(x="stars", y="is_open")
#df.plot.scatter(x="review_count", y="stars")

#ax = sns.heatmap(df)

X = df[["review_count", "stars"]]
y = df["is_open"]
trainX, valX, trainY, valY = train_test_split(X, y, random_state = 0)

model = BaggingClassifier()


model.fit(trainX, trainY)
valPreds = model.predict(valX)
print("Making predictions for the following:")
print(valY[0:20])
print("The predictions are")
print(valPreds[0:20])
scores = cross_val_score(model, X, y, scoring = "accuracy", cv = 3)
print(scores)

