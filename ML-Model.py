import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier




#def main():
    # Read project data file


df = pd.read_csv("Project 1 Data.csv")
# No missing values so no drop required

#Split the data into training set of 80% and test of 20% using stratified split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=845645211)
for train_index, test_index in split.split(df, df["Step"]):
    strat_train_set = df.loc[train_index].reset_index(drop=True)
    strat_test_set = df.loc[test_index].reset_index(drop=True)

train_y = strat_train_set["Step"]
strat_train_set = strat_train_set.drop(columns=["Step"], axis = 1)
test_y = strat_test_set["Step"]
strat_test_set = strat_test_set.drop(columns = ["Step"], axis = 1)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ay = fig.add_subplot()
# strat_train_set.plot(kind="scatter", x="X", y="Z", alpha=0.1)
# p = ax.scatter(strat_train_set["X"], strat_train_set["Y"], strat_train_set["Z"], c = train_y)
# fig.colorbar(p)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')


pd.plotting.scatter_matrix(strat_train_set, c = train_y)
plt.figure(0)


corr_matrix = strat_train_set.corr()
sns.heatmap(np.abs(corr_matrix), annot = True)


nearestn = KNeighborsClassifier(n_neighbors = 3)
nearestn.fit(strat_train_set, train_y)


nearest_train_prediction = nearestn.predict(strat_train_set)
nearestn_train_mae = sklearn.metrics.mean_absolute_error(nearest_train_prediction, train_y)

print("Model 1 training MAE is: ", round(nearestn_train_mae,2))

nearestn_test_prediction = nearestn.predict(strat_test_set)
nearestn_test_mae = sklearn.metrics.mean_absolute_error(nearestn_test_prediction, test_y)

print("Model 1 test set MAE is: ", round(nearestn_test_mae,2))

plt.figure(2)
plt.scatter(x = test_y, y = nearestn_test_prediction)
plt.show()


# param_grid = {
#     'n_estimators': [10, 30, 50],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['sqrt', 'log2']
# }
# model2 = RandomForestRegressor(random_state=42)
# grid_search = GridSearchCV(model2, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
# grid_search.fit(train_X, train_y)
# best_params = grid_search.best_params_
# print("Best Hyperparameters:", best_params)
# best_model2 = grid_search.best_estimator_


perceptron_model = MLPClassifier(random_state = 0, max_iter = 500000).fit(strat_train_set, train_y)

perceptron_train_prediction = perceptron_model.predict(strat_train_set)
perceptron_train_mae = sklearn.metrics.mean_absolute_error(perceptron_train_prediction, train_y)

print("Model 2 training MAE is: ", round(perceptron_train_mae,2))

perceptron_test_prediction = perceptron_model.predict(strat_test_set)
perceptron_test_mae = sklearn.metrics.mean_absolute_error(perceptron_test_prediction, test_y)

print("Model 2 test set MAE is: ", round(perceptron_test_mae,2))

plt.figure(3)
plt.scatter(x = test_y, y = perceptron_test_prediction)
plt.show()

print(perceptron_model.score(strat_test_set, test_y))




randfor_model = RandomForestClassifier(n_estimators=100).fit(strat_train_set, train_y)

randfor_model_prediction = randfor_model.predict(strat_train_set)
randfor_model_mae = sklearn.metrics.mean_absolute_error(randfor_model_prediction, train_y)

print("Model 3 training MAE is: ", round(randfor_model_mae,2))

randfor_test_prediction = randfor_model.predict(strat_test_set)
randfor_test_mae = sklearn.metrics.mean_absolute_error(randfor_test_prediction, test_y)

print("Model 3 test set MAE is: ", round(randfor_test_mae,2))

plt.figure(4)
plt.scatter(x = test_y, y = randfor_test_prediction)
plt.show()



print(randfor_model.score(strat_test_set, test_y))

#print(corr_matrix["Step"].sort_values())



#if __name__ == "__main__":
#    main()