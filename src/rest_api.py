import flask
from flask import Flask, request, redirect, make_response
from flask_cors import CORS
import json
import sys
import codecs
import logging
import urllib
import config
import tensorflow as tf
import sklearn
from sklearn.externals import joblib
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import load_model


# Python2.x: Make default encoding as UTF-8
if sys.version_info.major == 2:
    reload(sys)
    sys.setdefaultencoding('UTF8')


logging.basicConfig(filename='/tmp/error.log', level=logging.DEBUG)
app = Flask(__name__)
app.config.from_object('config')
CORS(app)


@app.route('/')
def heart_beat():
    return flask.jsonify({"status": "ok"})


@app.route('/api/v1/cfme_recommender', methods=['POST'])
def cfme_recommender():
    input_json = request.get_json(force=True)
    print "This is the input"
    print input_json
    
    #load the model
    print "\n Current path \n"
    print os.getcwd()
    print "\n"
    model_mem_threshold = joblib.load("/models/memoryThreshold.pkl")
    model_num_workers = joblib.load("/models/numWorkers.pkl")

    # score the payload
    payload_to_score = pd.DataFrame(
            {
                "base_worker_memory"      : [input_json['base_worker_memory']],
                "estimated_pss_memory"    : [input_json['estimated_pss_memory']],
                "maximum_pss_memory"      : [input_json['maximum_pss_memory']],
                "metrics_collection_time" : [input_json['metrics_collection_time']],
                "num_objects_to_collect"  : [input_json['num_objects_to_collect']],
                "percent_cpu_util"        : [input_json['percent_cpu_util']]
                })

    predicted_memory_threshold = model_mem_threshold.predict(payload_to_score)[0]
    predicted_num_workers = model_num_workers.predict(payload_to_score)[0]

    # create the response object
    response = {
            "recommended_memory_threshold": predicted_memory_threshold,
            "recommended_number_of_workers": predicted_num_workers
            }

    # return the response object
    return flask.jsonify(response)


if __name__ == "__main__":
    app.run()
