import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.externals import joblib

import generate_data

def train_model_decisiontree():
    """
      This function trains a decision tree model on the CFME data
      This function creates two models to predict:
        a) Number of workers needed to run the given workload
        b) Memory threshold for each of the workers to run the given workload
    """

    #generate the training data
    train = generate_data.generate_data()

    # Decision tree model

    #Instantiate the models
    model_num_workers = DecisionTreeRegressor()
    model_mem_threshold = DecisionTreeClassifier()

    #fit the models
    model_num_workers.fit(train.iloc[:,0:6], train.iloc[:, 7])
    model_mem_threshold.fit(train.iloc[:,0:6], train.iloc[:,6])

    #serialize the models so that it can be used by the REST API
    joblib.dump(model_mem_threshold, "models/memoryThreshold_decisiontree.pkl")
    joblib.dump(model_num_workers, "models/numWorkers_decisiontree.pkl")


if __name__ == "__main__":
    train_model_decisiontree()
