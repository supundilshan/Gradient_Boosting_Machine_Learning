from sklearn.model_selection import train_test_split
# evaluate gradient boosting ensemble for regression
from pandas import read_csv
# import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle

from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import GradientBoostingRegressor

# load dataset
Train_dataframe = read_csv("Input/Pure_Data.csv")
Train_dataset = Train_dataframe.values
X_train = Train_dataset[:, 3:10]
Y_train = Train_dataset[:, 2]

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

prediction = model.predict(X_train)
Y_Actual = Y_train

MSE = mean_squared_error(Y_Actual, prediction)
print("MSE = ", MSE)

# get R-squared
r2 = r2_score(Y_Actual, prediction)
print("R-squared : ", r2)

# get MAPE
MAPE = mean_absolute_percentage_error(Y_Actual, prediction)
print("MAPE : ", MAPE)

plt.plot(prediction, Y_train, '.', color='black')
# create scatter plot
m, b = np.polyfit(prediction, Y_train, 1)
# m = slope, b=intercept
plt.plot(prediction, m*prediction + b, color='red')
plt.xlabel("Predicted values")
plt.ylabel("Actual Values")


# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
