from pandas import read_csv
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

# load dataset
Train_dataframe = read_csv("Input/Train_Data.csv")
Train_dataset = Train_dataframe.values
X_train = Train_dataset[:, 3:10]
Y_train = Train_dataset[:, 2]

# load dataset
Test_dataframe = read_csv("Input/Test_Data.csv")
Test_dataset = Test_dataframe.values
X_test = Test_dataset[:, 3:10]
Y_test = Test_dataset[:, 2]

# load the model from disk
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)

prediction = loaded_model.predict(X_train)
Y_Actual = Y_train

MSE = mean_squared_error(Y_Actual, prediction)
print("MSE = ", MSE)

# get R-squared
r2 = r2_score(Y_Actual, prediction)
print("R-squared : ", r2)

# get MAPE
MAPE = mean_absolute_percentage_error(Y_Actual, prediction)
print("MAPE : ", MAPE)