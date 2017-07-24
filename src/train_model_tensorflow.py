import numpy as np
import pandas as pd
import tensorflow as tf
import tflearn
from sklearn.externals import joblib

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



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
    Y2 = np.array(pd.get_dummies(Y2))  # label: memory threshold

    # tensorflow model for memory threshold.
    # This model uses tflearn API

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
    model_mem_threshold.fit(X, Y2, n_epoch=25, show_metric=True, run_id="dense_model")


    # tensorflow model for number of workers
    # this uses keras API

     # create model
    model_num_workers = Sequential()
    model_num_workers.add(Dense(6, input_dim=6, kernel_initializer='normal', activation='relu'))
    model_num_workers.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model_num_workers.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model_num_workers.compile(loss='mean_squared_error', optimizer='adam')
    model_num_workers.fit(X, Y1, epochs=25, batch_size=16,verbose=1)

    #TODO: Point to note (Need to investigate later):
    # Running this code again and again may result in "list out of index" error.
    # The only way to come out of this seem to be killing the Python session and starting again
    # Also, note that both the tensorflow models severely overfit the given data
    # Need to tune the models to avoid over-fitting

    #serialize the models so that it can be used by the REST API
    model_mem_threshold.save("models/model_mem_threshold.tflearn")
    model_num_workers.save("models/model_num_workers.h5")


if __name__ == "__main__":
    train_model_tensorflow()
