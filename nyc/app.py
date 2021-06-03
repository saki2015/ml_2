from flask import Flask, render_template, request
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from utils import preprocess, feature_engg
from utils import seed_everything#, feature_engineer_test_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import random, os
from transformers import *
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import itertools

from sklearn.linear_model import (
    ElasticNetCV, LassoCV, RidgeCV, LinearRegression, Lasso, SGDRegressor
)
from sklearn.svm import LinearSVR
from sklearn.ensemble import (RandomForestRegressor, StackingRegressor)
from xgboost import XGBRegressor
from scipy.stats import skew
from lightgbm import LGBMRegressor

import pickle

app = Flask(__name__)


def feature_engineer_test_data(X, impute_dist=0):
    cat_pipe = pickle.load(open('catpipe.sav', 'rb'))
    num_pipe = pickle.load(open('numpipe.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))
    
    ctrans_df = cat_pipe.transform(X)
    print('Done with categorical transform')
    print('=====ctrans_df=======')
    print(ctrans_df)

    ctrans_df.drop(columns=['pickup_datetime'], inplace=True)

    ntrans_df = num_pipe.transform(X)
    print('Done with numeric transform')
    print('=====ntrans_df=======')

    print(ntrans_df)

    combined_df = pd.concat([ctrans_df, ntrans_df], axis=1)
    print('=====combined_df=======')

    print('combined_df shape: {}'.format(combined_df.shape))
    print(combined_df.info())
        
    trans_df = combined_df
    print('=====trans_df=======')

    print(trans_df)

    scaled_data = scaler.transform(trans_df)
    scaled_df = pd.DataFrame(scaled_data, columns=trans_df.columns, index=trans_df.index)

    print('Done with scaling')
    print(scaled_df)

    return scaled_df

def predict_for_test_data(model, test_df=None):
    seed_everything(42)
    if test_df is None:
        fpath = './test.csv'
        test_df = pd.read_csv(fpath, parse_dates=['pickup_datetime'])

        mtest_df = test_df.drop(['key'], axis=1)
        scaled_test_df= feature_engineer_test_data(mtest_df, impute_dist=0)
    else:
        print('user supplied data:')
        print(test_df)
        scaled_test_df= feature_engineer_test_data(test_df)
    print(scaled_test_df.columns)
    print(scaled_test_df)
    y_pred_orig = model.predict(scaled_test_df)
    print('y_pred_orig: {}'.format(y_pred_orig[0]))
    return y_pred_orig

@app.route("/predict", methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':
        src_lat = float(request.form['src_lat'])
        src_long = float(request.form['src_long'])
        dest_lat = float(request.form['dest_lat'])
        dest_long = float(request.form['dest_long'])

        pickup_date_time = pd.to_datetime(request.form['pick_date_time'])
        num_passengers = int(request.form['num_passengers'])
        inp_df = pd.DataFrame(
            {
                "pickup_longitude": src_long,
                "pickup_latitude": src_lat,
                "dropoff_longitude":dest_long,
                "dropoff_latitude": dest_lat,
                "pickup_datetime": pickup_date_time,
                "passenger_count": num_passengers,
            },
            index=[0],
        )
        print(inp_df)
        print(inp_df.info())
        saved_model = './nyc_stacked_model.sav'
        loaded_model = pickle.load(open(saved_model, 'rb'))
        y_pred = predict_for_test_data(loaded_model, inp_df)
        print('y_pred shape: {}'.format(y_pred.shape))
        return render_template('index.html', pred=str(y_pred[0]))

    else:
        return render_template('index.html')

def main():
    predict()

#main()
