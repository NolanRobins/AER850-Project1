import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import StratifiedShuffleSplit



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
#strat_test_set = strat_test_set.drop(columns=["Step"], axis = 1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ay = fig.add_subplot()

pd.plotting.scatter_matrix(strat_train_set, c = train_y)
#strat_train_set.plot(kind="scatter", x="X", y="Z", alpha=0.1)
p = ax.scatter(strat_train_set["X"], strat_train_set["Y"], strat_train_set["Z"], c = train_y)
fig.colorbar(p)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
#ay.scatter(strat_train_set["X"], strat_train_set["Y"], s=200, c=strat_train_set["Step"], cmap='Greens')




plt.show()
corr_matrix = strat_train_set.corr()
#print(corr_matrix["Step"].sort_values())



#if __name__ == "__main__":
#    main()