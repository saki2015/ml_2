import pandas as pd

import os
import numpy as np
import time
import statistics
from sklearn import metrics
from haversine import haversine, Unit
from transformers import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle

def seed_everything(seed=42):
    import random, os, numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def preprocess(df): 
    # remove missing values in the dataframe
    def remove_missing_values(df):
        df = df.dropna()
        return df
  
    # remove outliers in fare amount
    def remove_fare_amount_outliers(df, lower_bound, upper_bound):
        df = df[(df['fare_amount'] > lower_bound) & (df['fare_amount'] <= upper_bound)]
        return df
  
    # replace outliers in passenger count with the mode
    def replace_passenger_count_outliers(df):
        df.loc[df['passenger_count'] == 0, 'passenger_count'] = 1
        return df
  
    # remove outliers in latitude and longitude
    def remove_lat_long_outliers(df):
        # range of longitude for NYC
        
        nyc_min_longitude = -74.05
        nyc_max_longitude = -73.75

        # range of latitude for NYC
        nyc_min_latitude = 40.63
        nyc_max_latitude = 40.85
        
        # only consider locations within New York City
        for long in ['pickup_longitude', 'dropoff_longitude']:
            df = df[df[long].between(nyc_min_longitude, nyc_max_longitude)]
            
        for lat in ['pickup_latitude', 'dropoff_latitude']:
            df = df[df[lat].between(nyc_min_latitude, nyc_max_latitude)]

        return df

    df = remove_missing_values(df)
    df = remove_fare_amount_outliers(df, lower_bound = 0, upper_bound = 100)
    df = replace_passenger_count_outliers(df)
    df = remove_lat_long_outliers(df)
    return df

def feature_engg(X, impute_dist=0):   
    '''
    categorical_columns = X.select_dtypes(include=['datetime64[ns, UTC]']).columns
    numeric_columns = X.select_dtypes(exclude=['object', 'datetime64[ns, UTC]']).columns
    '''
    categorical_columns = pd.Index(['pickup_datetime']) 
    numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns

    print('categorical cols: {}'.format(categorical_columns))
    print('numeric cols: {}'.format(numeric_columns))
    
    cat_pipe = make_pipeline(
        ColumnExtractor(categorical_columns),
        #DFSimpleImputer(strategy='most_frequent'),
        DateTransformer('pickup_datetime'),
        CheckSpecialTimesTransformer(),
        )
    ctrans_df = cat_pipe.fit_transform(X)
    pickle.dump(cat_pipe, open('catpipe.sav', 'wb'))
    
    print('Done with categorical transform')
    print(ctrans_df)
    # Now that all info sifted from 'pickup_datetime' field, 
    # we can drop the field
    ctrans_df.drop(columns=['pickup_datetime'], inplace=True)
    
    print('Dropped date')
    num_pipe = make_pipeline(
        ColumnExtractor(numeric_columns),
        #DFSimpleImputer(strategy='mean'),
        CalcDistTransformer(), 
        AirportDistTransformer())
    
    ntrans_df = num_pipe.fit_transform(X)
    pickle.dump(num_pipe, open('numpipe.sav', 'wb'))

    print('Done with numeric transform')
    print(ntrans_df)
    # concatenate the 2 Dataframes from the 2 pipelines -categorical + numeric
    combined_df = pd.concat([ctrans_df, ntrans_df], axis=1)
    trans_df = combined_df
    
    '''
    if impute_dist:
        # Use KNNImputer to impute 'hdist_mi' field with <= 0 mile distance
        print('Proceeding to impute distance...')
        trans_df = process_dist_outliers(combined_df, 0)
    else:
        print('no imputing ...')   
    '''
    
    print('scaling ...')   
    # scale the data
    s = StandardScaler()
    scaled = s.fit_transform(trans_df)
    pickle.dump(s, open('scaler.sav', 'wb'))
    
    print('trans_df shape: {}'.format(trans_df.shape))
    # Convert numpy array from scaling to a DataFrame
    scaled_df = pd.DataFrame(scaled, columns = trans_df.columns, index=trans_df.index)
    print('scaled_df shape: {}'.format(scaled_df.shape))
    print('=======after scaling===========')
    print(scaled_df.head(2))
    
    return scaled_df

def feature_engineer_test_data(X, impute_dist=0):
    cat_pipe = pickle.load(open('catpipe.sav', 'rb'))
    num_pipe = pickle.load(open('numpipe.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))
    
    #categorical_columns = pd.Index(['pickup_datetime']) 
    #numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
    
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
