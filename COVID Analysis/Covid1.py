# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 21:57:28 2020

@author: HEMANTH
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import the dataset
dataset = pd.read_csv('Covid Dataset.csv')
X = dataset.iloc[:,1:21].values
Y = dataset.iloc[:,20].values

#Data Preprocessing
#Using Encooding technique
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])
labelencoder_X_4 = LabelEncoder()
X[:, 4] = labelencoder_X_4.fit_transform(X[:, 4])

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(rmse)
y_pred = list(y_pred)

for i in range(len(y_pred)):
    if y_pred[i] ==1:
        y_pred[i] = "Positive"
    else:
        y_pred[i]= "Negative"
