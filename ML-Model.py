import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import joblib





def run_model(model, parameter_grid, training_set_x, training_set_y, test_set_x, test_set_y, model_name):

    model_grid_search = GridSearchCV(model, parameter_grid, cv = 5, scoring = 'neg_mean_absolute_error', n_jobs = -1)

    model_grid_search.fit(training_set_x, training_set_y)

    best_model = model_grid_search.best_estimator_

    training_prediction = best_model.predict(training_set_x)
    train_mae = sklearn.metrics.mean_absolute_error(training_prediction, training_set_y)

    test_prediction = best_model.predict(test_set_x)
    test_mae = sklearn.metrics.mean_absolute_error(test_prediction, test_set_y)

    plt.figure()
    plt.scatter(x = training_set_y, y = training_prediction)
    plt.scatter(x = test_set_y, y = test_prediction)
    plt.show()


    print("Best Hyperparameters For", model_name, ":", model_grid_search.best_params_)
    print("Model", model_name, "training MAE is: ", round(train_mae, 3))
    print("Model", model_name, "test set MAE is: ", round(test_mae, 3))


    model_recall_score = recall_score(y_true = test_set_y, y_pred = test_prediction, average = 'micro')
    print("Model", model_name, "recall score is: ", round(model_recall_score, 3))

    model_precision_score = precision_score(y_true = test_set_y, y_pred = test_prediction, average = 'micro')
    print("Model", model_name, "precision score is: ", round(model_precision_score, 3))

    model_f1_score = f1_score(y_true = test_set_y, y_pred = test_prediction, average = 'micro')
    print("Model", model_name, "f1 score is: ", round(model_f1_score, 3))

    return [best_model, test_prediction]

# def main():
    # Read project data file


df = pd.read_csv("Project 1 Data.csv")
# No missing values so no drop required

# Split the data into training set of 80% and test of 20% using stratified split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=845645211)
for train_index, test_index in split.split(df, df["Step"]):
    strat_train_set = df.loc[train_index].reset_index(drop=True)
    strat_test_set = df.loc[test_index].reset_index(drop=True)

train_y = strat_train_set["Step"]
strat_train_set = strat_train_set.drop(columns=["Step"], axis = 1)
test_y = strat_test_set["Step"]
strat_test_set = strat_test_set.drop(columns = ["Step"], axis = 1)

data_scaler = preprocessing.StandardScaler().fit(strat_train_set)
train_x = pd.DataFrame(data_scaler.transform(strat_train_set))
test_x = pd.DataFrame(data_scaler.transform(strat_test_set))


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ay = fig.add_subplot()
# strat_train_set.plot(kind="scatter", x="X", y="Z", alpha=0.1)
# p = ax.scatter(strat_train_set["X"], strat_train_set["Y"], strat_train_set["Z"], c = train_y)
# fig.colorbar(p)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')


pd.plotting.scatter_matrix(train_x, c = train_y)
plt.figure()


corr_matrix = train_x.corr()
sns.heatmap(np.abs(corr_matrix), annot = True)


#------------------------------------------------Nearest Neighbor-----------------------------------------------

nearest_n = KNeighborsClassifier()


nearest_n_param_grid = {
    'n_neighbors': [2, 3, 5, 10, 25, 50],
    'weights': ['uniform', 'distance'],
    'leaf_size': [5, 10, 20, 30, 40, 50],
    'algorithm': ['ball_tree', 'kd_tree'],
}

[nearest_n, nearest_n_test_prediction] = run_model(nearest_n, nearest_n_param_grid, train_x, train_y, test_x, test_y, 'Nearest Neighbor')

#---------------------------------------------------------------------------------------------------------------


#---------------------------------------------Multi-layer Perceptron--------------------------------------------

perceptron_model = MLPClassifier(random_state = 0, max_iter = 500000)


perceptron_model_param_grid = {
    'hidden_layer_sizes': [10, 50, 100, 200, 500],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['adam', 'lbfgs'],
}

_ = run_model(perceptron_model, perceptron_model_param_grid, train_x, train_y, test_x, test_y, 'Multi-layer Perceptron')

#---------------------------------------------------------------------------------------------------------------


#-------------------------------------------------Random Forest-------------------------------------------------

rand_for_model = RandomForestClassifier(random_state = 5)


rand_for_model_param_grid = {
    'n_estimators': [10, 50, 100, 200, 500],
    'min_samples_leaf': [0.5, 1, 2, 5, 10, 20],
    'max_features': ['sqrt', 'log2', None],
}

_ = run_model(rand_for_model, rand_for_model_param_grid, train_x, train_y, test_x, test_y, 'Random Forest')

#---------------------------------------------------------------------------------------------------------------


nearest_n_confusion_matrix = confusion_matrix(y_true = test_y, y_pred = nearest_n_test_prediction)
confusion_disp = ConfusionMatrixDisplay(confusion_matrix=nearest_n_confusion_matrix)
confusion_disp.plot()
plt.show()

joblib.dump(nearest_n, 'nearest_n_model.joblib')

loaded_model = joblib.load('nearest_n_model.joblib')

to_predict = pd.DataFrame([[9.375, 3.0625, 1.51], [6.995, 5.125, 0.3875], [0,3.0625, 1.93], [9.4, 3, 1.8], [9.4, 3, 1.3]])
to_predict.columns = ['X', 'Y', 'Z']

predictions = loaded_model.predict(pd.DataFrame(data_scaler.transform(to_predict)))

print(predictions)


#if __name__ == "__main__":
#    main()