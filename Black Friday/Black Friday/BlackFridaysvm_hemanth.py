
# Basic libraries for data manipulation and analysis
import pandas as pd
import numpy as np

BlackFridayTrain = pd.read_csv('train.csv')
BlackFridayTest= pd.read_csv('test.csv')

# Replace missing values in Product_Category_2 and Product_Category_3 columns with 0
# because replacing these missing values with mean/median/mode might introduce bias in the data
BlackFridayTrain['Product_Category_2'].fillna(0, inplace=True)
BlackFridayTrain['Product_Category_3'].fillna(0, inplace=True)
# Converting Product_Category_2 and Product_Category_3 to int from object, to maintain consistency across the dataframe
BlackFridayTrain['Product_Category_2'] = BlackFridayTrain['Product_Category_2'].astype(int)
BlackFridayTrain['Product_Category_3'] = BlackFridayTrain['Product_Category_3'].astype(int)

BlackFridayTrainNew = BlackFridayTrain.copy(deep=True)

# Replacing column values in the dataframe to maintain consistency throughout

# In Gender, replacing 'F' with 0 and 'M' with 1
BlackFridayTrainNew.loc[BlackFridayTrainNew['Gender'] == 'F', 'Gender'] = 0
BlackFridayTrainNew.loc[BlackFridayTrainNew['Gender'] == 'M', 'Gender'] = 1

# In Age column, replacing different ranges with the below values
# '0-17' is replaced with 0
# '18-25' is replaced with 1
# '26-35' is replaced with 2
# '36-45' is replaced with 3
# '46-50' is replaced with 4
# '51-55' is replaced with 5
# '55+' is replaced with 6
BlackFridayTrainNew.loc[BlackFridayTrainNew['Age'] == '0-17', 'Age'] = 0
BlackFridayTrainNew.loc[BlackFridayTrainNew['Age'] == '18-25', 'Age'] = 1
BlackFridayTrainNew.loc[BlackFridayTrainNew['Age'] == '26-35', 'Age'] = 2
BlackFridayTrainNew.loc[BlackFridayTrainNew['Age'] == '36-45', 'Age'] = 3
BlackFridayTrainNew.loc[BlackFridayTrainNew['Age'] == '46-50', 'Age'] = 4
BlackFridayTrainNew.loc[BlackFridayTrainNew['Age'] == '51-55', 'Age'] = 5
BlackFridayTrainNew.loc[BlackFridayTrainNew['Age'] == '55+', 'Age'] = 6

# In 'Stay_In_Current_City_Years', replacing '4+' with 4
BlackFridayTrainNew.loc[BlackFridayTrainNew['Stay_In_Current_City_Years'] == '4+', 'Stay_In_Current_City_Years'] = 4

# In 'City_Category',replacing different categories with the values listed below;
# 'A' is replaced with 0
# 'B' is replaced with 1
# 'C' is replaced with 2
BlackFridayTrainNew.loc[BlackFridayTrainNew['City_Category'] == 'A', 'City_Category'] = 0
BlackFridayTrainNew.loc[BlackFridayTrainNew['City_Category'] == 'B', 'City_Category'] = 1
BlackFridayTrainNew.loc[BlackFridayTrainNew['City_Category'] == 'C', 'City_Category'] = 2

# Converting all object type columns to integer to maintain consistency across the dataframe
BlackFridayTrainNew['Gender'] = BlackFridayTrainNew['Gender'].astype(int)
BlackFridayTrainNew['Age'] = BlackFridayTrainNew['Age'].astype(int)
BlackFridayTrainNew['City_Category'] = BlackFridayTrainNew['City_Category'].astype(int)
BlackFridayTrainNew['Stay_In_Current_City_Years'] = BlackFridayTrainNew['Stay_In_Current_City_Years'].astype(int)


cols = list(BlackFridayTrainNew)

# move the column to head of list using index, pop and insert
cols.insert(14, cols.pop(cols.index('Purchase')))
BlackFridayTrainNew = BlackFridayTrainNew.loc[:, cols]

#For future reference save it in csv
BlackFridayTrainNew.to_csv('BlackFridayTrainNew.csv')
# corr indicates the correlation between all the features of the dataset
# It can be seen that the newly introduced feautres 'Category_Count', 'User_Score' and 'Product_Score'
# have higher correlation scores with target variable (purchase amount) than other features
correlation_var=BlackFridayTrainNew.corr()


 # FOR TEST DATA

# Replace missing values in Product_Category_2 and Product_Category_3 columns with 0
#  because replacing these missing values with mean/median/mode might introduce bias in the data
BlackFridayTest['Product_Category_2'].fillna(0, inplace=True)
BlackFridayTest['Product_Category_3'].fillna(0, inplace=True)
# Converting Product_Category_2 and Product_Category_3 to int from object, to maintain consistency across the dataframe
BlackFridayTest['Product_Category_2'] = BlackFridayTest['Product_Category_2'].astype(int)
BlackFridayTest['Product_Category_3'] = BlackFridayTest['Product_Category_3'].astype(int)

BlackFridayTestNew = BlackFridayTest.copy(deep=True)

# Replacing column values in the dataframe to maintain consistency throughout

# In Gender, replacing 'F' with 0 and 'M' with 1
BlackFridayTestNew.loc[BlackFridayTrainNew['Gender'] == 'F', 'Gender'] = 0
BlackFridayTestNew.loc[BlackFridayTrainNew['Gender'] == 'M', 'Gender'] = 1

#  In Age column, replacing different ranges with the below values
#  '0-17' is replaced with 0
#  '18-25' is replaced with 1
#  '26-35' is replaced with 2
#  '36-45' is replaced with 3
#  '46-50' is replaced with 4
#  '51-55' is replaced with 5
#  '55+' is replaced with 6
BlackFridayTestNew.loc[BlackFridayTestNew['Age'] == '0-17', 'Age'] = 0
BlackFridayTestNew.loc[BlackFridayTestNew['Age'] == '18-25', 'Age'] = 1
BlackFridayTestNew.loc[BlackFridayTestNew['Age'] == '26-35', 'Age'] = 2
BlackFridayTestNew.loc[BlackFridayTestNew['Age'] == '36-45', 'Age'] = 3
BlackFridayTestNew.loc[BlackFridayTestNew['Age'] == '46-50', 'Age'] = 4
BlackFridayTestNew.loc[BlackFridayTestNew['Age'] == '51-55', 'Age'] = 5
BlackFridayTestNew.loc[BlackFridayTestNew['Age'] == '55+', 'Age'] = 6

 # In 'Stay_In_Current_City_Years', replacing '4+' with 4
BlackFridayTestNew.loc[BlackFridayTestNew['Stay_In_Current_City_Years'] == '4+', 'Stay_In_Current_City_Years'] = 4

 # In 'City_Category',replacing different categories with the values listed below;
 # 'A' is replaced with 0
 # 'B' is replaced with 1
 # 'C' is replaced with 2
BlackFridayTestNew.loc[BlackFridayTestNew['City_Category'] == 'A', 'City_Category'] = 0
BlackFridayTestNew.loc[BlackFridayTestNew['City_Category'] == 'B', 'City_Category'] = 1
BlackFridayTestNew.loc[BlackFridayTestNew['City_Category'] == 'C', 'City_Category'] = 2

 # Converting all object type columns to integer to maintain consistency across the dataframe
df_Gender = pd.get_dummies(BlackFridayTestNew['Gender'])
BlackFridayTestNew['Gender'] = df_Gender
BlackFridayTestNew['Age'] = BlackFridayTestNew['Age'].astype(int)
BlackFridayTestNew['City_Category'] = BlackFridayTrainNew['City_Category'].astype(int)
BlackFridayTestNew['Stay_In_Current_City_Years'] = BlackFridayTestNew['Stay_In_Current_City_Years'].astype(int)





BlackFridayTestNew = BlackFridayTestNew.drop(['Product_ID'],axis=1)
#For future reference save it in csv
BlackFridayTestNew.to_csv('BlackFridayTestNew.csv')
# corr indicates the correlation between all the features of the dataset
# It can be seen that the newly introduced feautres 'Category_Count', 'User_Score' and 'Product_Score'
# have higher correlation scores with target variable (purchase amount) than other features
correlation_var=BlackFridayTestNew.corr()



X = BlackFridayTrainNew.drop(['Product_ID'], axis=1)
y = BlackFridayTrainNew.iloc[:,-1].values
y= np.reshape(y,(1,-1))
#  importing the test dataset
BlackFridayTestNew = pd.read_csv('BlackFridayTestNew.csv')
BlackFridayTestNew = BlackFridayTestNew.drop(['Product_ID'], axis=1)
x_test = BlackFridayTestNew.iloc[:,1:]



# from sklearn.svm import SVR
# regressor = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
# regressor.fit(X,y.ravel())
# y_pred= regressor.predict(x_test)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y.ravel())

# Predicting a new result
y_pred = regressor.predict(x_test)




from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y, y_pred))
print(rmse)














