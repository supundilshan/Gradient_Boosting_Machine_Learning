# example of grid searching key hyperparameters for gradient boosting on a classification dataset
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from pandas import read_csv
from sklearn.model_selection import train_test_split

# import sklearn
# scores = sorted(sklearn.metrics.SCORERS.keys())
# print(scores)

# load dataset
Train_dataframe = read_csv("Input/Train_Data.csv")
Train_dataset = Train_dataframe.values
X_train = Train_dataset[:, 3:10]
Y_train = Train_dataset[:, 2]

# # define the model with default hyperparameters
# # model = GradientBoostingClassifier()
model = GradientBoostingRegressor()

# define the grid of values to search
# number of trees
n_estimators = [100, 90, 80, 70, 50,40,30,10]
learning_rate = [0.1, 0.001, 0.0001,0.0001]
subsample = [1, 0.9, 0.8,0.7,0.6,0.5]
max_depth = [4,3, 2,1]
# min_samples_leaf=[1,2]
# min_samples_split=[2,3]

param_grid = dict(n_estimators=n_estimators, learning_rate=learning_rate, subsample=subsample, max_depth=max_depth)

# define the evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='neg_mean_squared_error', verbose=2)
# execute the grid search
grid_result = grid_search.fit(X_train, Y_train)

# summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# summarize all scores that were evaluated
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))