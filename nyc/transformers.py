#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

from haversine import haversine, Unit
from sklearn.preprocessing import MinMaxScaler

def check_after_hrs(dayOfWeek, hr):
        if dayOfWeek < 5 and (hr>=20 or hr <=6):
            return 1
        return 0  
    
def check_weekend(day):
    return 1 if day > 4 else 0

def calc_distance_in_miles(plat, plong, dlat, dlong):
        ptuple = (plat, plong)
        dtuple = (dlat, dlong)
        return haversine(ptuple, dtuple, unit='mi')
    
def calc_travel_dist(df):
    df['hdist_mi'] = df.apply(lambda x: calc_distance_in_miles(
                                x.pickup_latitude,
                                x.pickup_longitude,
                                x.dropoff_latitude,
                                x.dropoff_longitude), axis=1) 
    return df

class CheckSpecialTimesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        df['is_weekend'] = df['day_of_week'].apply(check_weekend)
        df['is_after_hrs'] =\
            df.apply(
                lambda x: check_after_hrs(x['day_of_week'], x['hour']), 
                axis = 1)
        return df
    
class DateTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, variable):
        self.variable = variable
        
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        df['year'] = df[self.variable].dt.year
        df['month'] = df[self.variable].dt.month
        df['day'] = df[self.variable].dt.day
        df['day_of_week'] = df[self.variable].dt.dayofweek
        df['hour'] = df[self.variable].dt.hour
        return df

class CalcDistTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
        df['hdist_mi'] = df.apply(lambda x: calc_distance_in_miles(
                                x.pickup_latitude,
                                x.pickup_longitude,
                                x.dropoff_latitude,
                                x.dropoff_longitude), axis=1) 
        return df
 
  
class AirportDistTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, df, y=None):
        return self
    
    def transform(self, df):
         airports = {
             'jfk':(-73.78,40.643),
             'lag': (-73.87, 40.77),
             'new': (-74.18, 40.69)
         }
         for key, tuple in airports.items():
            df['pickup_dist_from_' + key] = df.apply(lambda x: calc_distance_in_miles(
                                    x.pickup_latitude,
                                    x.pickup_longitude,
                                    tuple[1], tuple[0]), axis=1)
            
            df['dropoff_dist_from_' + key] = df.apply(lambda x: calc_distance_in_miles(
                                    x.dropoff_latitude,
                                    x.dropoff_longitude,
                                    tuple[1], tuple[0]), axis=1)
        
         return df
     
from sklearn.impute import SimpleImputer

# use : mtmp = DFSimpleImputer(strategy='constant', fill_value=2).fit_transform(tmp)
class DFSimpleImputer(SimpleImputer):
    
    def __init__(self, missing_values=np.nan, strategy='mean', fill_value=1):
        super(DFSimpleImputer, self).__init__(missing_values=missing_values, strategy=strategy, fill_value=fill_value)
       
    def fit(self, X, y=None):
        super(DFSimpleImputer, self).fit(X)
        return self
    
    def transform(self, X):
        Xim = super(DFSimpleImputer, self).transform(X)
        Xim = pd.DataFrame(Xim, index=X.index, columns=X.columns)
        return Xim
  
class ColumnExtractor(TransformerMixin):
    
    def __init__(self, cols):
        self.cols = cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        Xcols = X[self.cols]
        return Xcols
