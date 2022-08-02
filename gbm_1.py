from sklearn.model_selection import train_test_split
# evaluate gradient boosting ensemble for regression
from pandas import read_csv
# import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
import numpy as np
import pickle

from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import StandardScaler

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

# ===== Standerdize Data set =====
# created scaler
# scaler = StandardScaler()
#
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
#
# scaler.fit(X_test)
# X_test = scaler.transform(X_test)

# training_data, testing_data = train_test_split(dataset, test_size=0.4, random_state=25)
#
# print(f"No. of training examples: {training_data.shape[0]}")
# print(f"No. of testing examples: {testing_data.shape[0]}")
#
# # split into input (X) and output (Y) variables
# X_test = testing_data[:, 3:8]
# Y_test = testing_data[:, 2]
# X_train = training_data[:, 3:8]
# Y_train = training_data[:, 2]

# define the model
# model = GradientBoostingRegressor(n_estimators = 50, learning_rate = 0.1, subsample = 1, max_depth = 6)
model = GradientBoostingRegressor()
# define the evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model
n_scores = cross_val_score(model, X_train, Y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
# report performance
print('MSE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

# fit the model on the whole dataset
model.fit(X_train, Y_train)
# make a single prediction
prediction = model.predict(X_train)
Y_Actual = Y_train
# summarize prediction
# print(yhat[1])

# get MSE manualy Method-2
# SUM_of_squired_error = np.sum(np.square(Y_train - prediction))
# Calculated_MSE = SUM_of_squired_error/600
# print("Calculated_MSE : ", Calculated_MSE)
MSE = mean_squared_error(Y_Actual, prediction)
print("MSE = ", MSE)

# get R-squared
r2 = r2_score(Y_Actual, prediction)
print("R-squared : ", r2)

# get MAPE
MAPE = mean_absolute_percentage_error(Y_Actual, prediction)
print("MAPE : ", MAPE)

# plt.plot(prediction, Y_Actual, 'o')
# # create scatter plot
# m, b = np.polyfit(prediction, Y_Actual, 1)
# # m = slope, b=intercept
# plt.plot(prediction, m*prediction + b)
# plt.xlabel("Predicted values")
# plt.ylabel("Actual Values")

# plt.plot(Y_train, color = 'red', label = 'Real data')
# plt.plot(prediction, color = 'blue', label = 'Predicted data')
# plt.title('Prediction')
# plt.legend()
# plt.show()

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
