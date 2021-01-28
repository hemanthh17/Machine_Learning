
# Basic libraries for data manipulation and analysis
import pandas as pd
import numpy as np

BlackFridayTrain = pd.read_csv('train.csv')

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

# Introducing a new feature 'Category_Count', which identifies the number of unique categories from each row in the dataframe,
# which indicates the number of unique categories a customer has shopped from
conditions = [
    (BlackFridayTrainNew['Product_Category_1'] != 0) & (BlackFridayTrainNew['Product_Category_2'] == 0) & (BlackFridayTrainNew['Product_Category_3'] == 0),
    (BlackFridayTrainNew['Product_Category_1'] != 0) & (BlackFridayTrainNew['Product_Category_2'] != 0) & (BlackFridayTrainNew['Product_Category_3'] == 0),
    (BlackFridayTrainNew['Product_Category_1'] != 0) & (BlackFridayTrainNew['Product_Category_2'] != 0) & (BlackFridayTrainNew['Product_Category_3'] != 0)]
choices = [1, 2, 3]
BlackFridayTrainNew['Category_Count'] = np.select(conditions, choices, default=0)




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

X = BlackFridayTrainNew.drop(['Purchase','Product_ID'], axis=1)
y = BlackFridayTrainNew['Purchase']
# importing the test dataset
# BlackFridayTestNew = pd.read_csv('test.csv')
# x_test = BlackFridayTestNew.drop(['Product_ID'], axis=1)
# x_test = BlackFridayTestNew.iloc[:,1:]

# #Cut OFF ABOVE SNIPPET AND TRY FOR VARIETY MODELS

# import tensorflow as tf

# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

# NN_model = Sequential()

# # The Input Layer :
# NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X.shape[1], activation='relu'))

# # The Hidden Layers :
# NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
# NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
# NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))

# # The Output Layer :
# NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))

# # Compile the network :
# NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

# checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
# checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
# callbacks_list = [checkpoint]
# NN_model.fit(X, y, epochs=1, batch_size=16, validation_split = 0.2, callbacks=callbacks_list)
# predictions = NN_model.predict(x_test)
# finalsolution = BlackFridayTestNew[['User_ID','Product_ID']]
# finalsolution['Purchase'] = predictions
# finalsolution.to_csv('finalsolution_TensorFlow.csv')

        