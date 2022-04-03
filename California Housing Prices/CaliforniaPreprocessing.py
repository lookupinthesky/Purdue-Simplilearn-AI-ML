from joblib import load
import pandas as pd
import numpy as np


class Preprocessing:
    """     Script for Preprocessing and predicting on test(new) data

    1. Mean Imputation
    2. Dummy Encoding
    3. GMM clusters of latitude and longitude
    4. Feature Combination - total_rooms, total_bedrooms, households and population
    5. latitude,longitude to x,y,z
    6. Scale using Training data parameters

    """
    def __init__(self):

        self.train_params = pd.read_csv("train_params.csv")
        self.train_params.set_index("Unnamed: 0", inplace=True)
        # Load various trained models
        self.latmodel = load("models/gmmlat.joblib")
        self.longmodel = load("models/gmmlong.joblib")
        self.scaler = load("models/standardscaler.joblib")
        self.linear_model = load("models/linearmodel.joblib")
    
    def transform(self, X):
        """
        Transforms the given feature matrix with a series of operations performed on the training set. 
        """
        X_na = self.imputenans(X)
        X_ohe = self.onehotencoding(X_na)
        X_gmm = self.gaussian_mixture_prep(X_ohe,self.latmodel, self.longmodel)
        X_feat = self.feature_combination(X_gmm)
        X_xyz = self.latlongtoxyz(X_feat)
        X_std = self.scaler.transform(X_xyz)
        return X_std

    def imputenans(self, X):
        
        """
        imputes missing values in total_bedrooms with mean of the training data
        """
        
        X['total_bedrooms'].fillna(self.train_params.loc['mean','total_bedrooms'], inplace=True)
        
        return X
        
        
    
    def gaussian_mixture_prep(self, X, gmmlat, gmmlong):

        """
        Creates new binary features by assigning cluster labels to latitude and longitude feature values calculated by 
        gaussian mixture model. 
        
        @params:
        gmmlat: gaussian mixture model for latitude
        gmmlong: gaussian mixture model for longitude
        
        """

        X['latitude_class'] = gmmlat.predict(X['latitude'].values.reshape(-1, 1))
        X['longitude_class'] = gmmlong.predict(X['longitude'].values.reshape(-1, 1))
        X['latitude_class'] = X['latitude_class'].astype(str)
        X['longitude_class'] = X['longitude_class'].astype(str)
        X['latlongcluster'] = X['latitude_class'] + X['longitude_class']
        X = X.drop(columns=['latitude_class', 'longitude_class'])
        X= pd.get_dummies(X, columns=['latlongcluster'])
        X.drop(columns=['latlongcluster_00'], inplace=True)
    
        return X

    def latlongtoxyz(self, X):
        
        """
        Creates new features by transforming latitude and longitude to x,y,z coordinates
        """
        
        X['xcoordinate'] = np.cos(X['latitude'])*np.cos(X['longitude'])
        X['ycoordinate'] = np.cos(X['latitude'])*np.sin(X['longitude'])
        X['zcoordinate'] = np.sin(X['latitude'])
        X = X.drop(columns = ['latitude','longitude'])
        
        return X
        
    def feature_combination(self, X):
        
        """
        creates new features using total_rooms, total_bedrooms, households and populatiom
        """
        
        X['total_rooms_per_person'] = X['total_rooms']/X['population']
        X['total_bedrooms_per_person'] = X['total_bedrooms']/X['population']
        X['households_per_person'] = X['households']/X['population']
        
        return X
                           
    def onehotencoding(self, X):
        
        """
        performs dummy encoding for the ocean proximity feature
        """
        
        X = pd.get_dummies(X, columns=['ocean_proximity'])
        X.drop(columns=['ocean_proximity_ISLAND'], inplace=True)
        return X
                           
    
    def predict(self, X):
        
        """
        provides predictions y for a given X
        """
        
        self.y_pred = self.linear_model.predict(X)
        return self.y_pred
                           