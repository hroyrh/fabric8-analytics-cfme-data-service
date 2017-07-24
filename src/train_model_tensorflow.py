import numpy as np
import pandas as pd
import tensorflow as tf
import tflearn
from sklearn.externals import joblib

import generate_data

def train_model_tensorflow():
    """
      This function trains a tensorflow model on the CFME data
      This function creates two models to predict:
        a) Number of workers needed to run the given workload
        b) Memory threshold for each of the workers to run the given workload

    """

    #generate the training data
    train = generate_data.generate_data()

    # tensorflow MLP model

    # Data for tensorflow model
    X  = np.array(train.iloc[:,0:6])  # input features
    Y1 = np.array(train.iloc[:, 7])   # label: number of workers

    #For categorical, labels need to be one-hot encoded for tflearn
    #Convert the column to string, run one-hot encoding and have it saved as a numpy array
    Y2 = pd.DataFrame(train.iloc[:, 6])
    Y2.worker_memory_threshold = Y2["worker_memory_threshold"].apply(str)
    Y2 = np.array(pd.get_dummies(Y2))
    Y2 = np.array(train.iloc[:, 6])   # label: memory threshold

    # Building deep neural network for memory threshold

    #This code is based on the MLP code available at: https://github.com/tflearn/tflearn/blob/master/examples/images/dnn.py

    #There are 6 input features. This need to change based on number of input features
    input_layer = tflearn.input_data(shape=[None, 6])
    dense1 = tflearn.fully_connected(input_layer, 64, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
    
    #Need to use dropouts. Need to tune dropout parameter. Without dropouts, model is severely overfitting
    dropout1 = tflearn.dropout(dense1, 0.5)
    dense2 = tflearn.fully_connected(dropout1, 2, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
    
    dropout2 = tflearn.dropout(dense2, 0.5)
    softmax = tflearn.fully_connected(dropout2, 2, activation='softmax')

    # Using SGD with learning rate decay and Top-3 accuracy
    sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
    top_k = tflearn.metrics.Top_k(3)
    net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                         loss='categorical_crossentropy')

    # Training
    model_mem_threshold = tflearn.DNN(net, tensorboard_verbose=0)

    #TODO: Point to note (Need to investigate later):
    # Running this code again and again may result in "list out of index" error.
    # The only way to come out of this seem to be killing the Python session and starting again

    model.fit(X, Y2, n_epoch=25, show_metric=True, run_id="dense_model")

    #serialize the models so that it can be used by the REST API
    joblib.dump(model_mem_threshold, "models/memoryThreshold_tensorflow.pkl")
    joblib.dump(model_num_workers, "models/numWorkers_tensorflow.pkl")


if __name__ == "__main__":
    train_model_tensorflow()
