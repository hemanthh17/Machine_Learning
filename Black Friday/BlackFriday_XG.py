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
y = dataset.iloc[:,-1].values
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# sc_y = StandardScaler()
# X = sc_X.fit_transform(X)
# y = sc_y.fit_transform(y.reshape(-1,1))


from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, -3:])
X[:,-3:] = imputer.transform(X[:, -3:])

imputer_new = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer_new = imputer.fit(y.reshape(-1,1))
y= imputer.transform(y.reshape(-1,1))
y=y.ravel()

X= (X - np.mean(X))/np.var(X)
y = (y- np.mean(y))/np.var(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.33, random_state=42)


from xgboost import XGBRegressor
regressor = XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.001,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(rmse)

# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)


# from sklearn.model_selection import cross_val_score
# accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
# print(accuracies.mean())
# print(accuracies.std())

df = pd.DataFrame(y_pred)
df.to_csv('out_hemanthXG.csv', index=False)
# # y_pred=regressor.predict(X_test)
# # df = pd.DataFrame(y_pred,X_test[:,0],X_test[:,1])
# # df.to_csv('out_hemanthsvm22.csv', index=False)


# X_grid = np.arange(min(X), max(X), 0.01) #this step required because data is feature scaled.
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(X_train, y_train, color = 'red')
# plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
# plt.title('Truth or Bluff (SVR)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
#plt.show()












