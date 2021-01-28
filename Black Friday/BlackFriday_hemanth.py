import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




dataset = pd.read_csv('test.csv')
#Error from here
df_Gender = pd.get_dummies(dataset['Gender'])
df_Age = pd.get_dummies(dataset['Age'])
df_City_Category = pd.get_dummies(dataset['City_Category'])
dfcurr_City_Category = pd.get_dummies(dataset['Stay_In_Current_City_Years'])
dataset['Stay_In_Current_City_Years'] = dfcurr_City_Category 
dataset['Gender'] = df_Gender
dataset['City_Category'] = df_City_Category
dataset['Age'] = df_Age




dataset=dataset.drop(['Product_ID'], axis=1)

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values.reshape(-1,1)

from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, -3:])
X[:,-3:] = imputer.transform(X[:, -3:])

imputer_new = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer_new = imputer.fit(y.reshape(-1,1))
y= imputer.transform(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred=regressor.predict(X_test)

# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(rmse)

df = pd.DataFrame(y_pred)

df.to_csv('out_hemanth.csv', index=False)












