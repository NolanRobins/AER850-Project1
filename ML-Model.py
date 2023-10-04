import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import StratifiedShuffleSplit



def main():
    # Read project data file
    df = pd.read_csv("Project 1 Data.csv")
    # No missing values so no drop required

    pd.plotting.scatter_matrix(df)
    df.plot(kind="scatter", x="X", y="Z", alpha=0.1)
    plt.show()
    corr_matrix = df.corr()
    print(corr_matrix["Step"].sort_values())


    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=845645211)
    for train_index, test_index in split.split(df, df["Step"]):
        strat_train_set = df.loc[train_index].reset_index(drop=True)
        strat_test_set = df.loc[test_index].reset_index(drop=True)
    strat_train_set = strat_train_set.drop(columns=["Step"], axis = 1)
    strat_test_set = strat_test_set.drop(columns=["Step"], axis = 1)



if __name__ == "__main__":
    main()