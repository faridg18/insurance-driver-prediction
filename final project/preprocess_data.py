# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:41:02 2017

@author: Jerry C
"""

import numpy as np
from sklearn import preprocessing
import math

def preprocess(x_train, y_train, x_validation):
    
    x_train, x_validation = add_missing_column_features(x_train, x_validation)
    #x_train, y_train = balance_data_by_duplicating_y_train(x_train, y_train) #balances dataset by downscaling
    
    x_train = change_missing_values_to_mean(x_train)
    x_validation = change_missing_values_to_mean(x_validation)
    
    
    #Normalization
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_validation = scaler.transform(x_validation)
    
    return x_train, y_train, x_validation

#Replaces missing values (values with -1) to the mean of that column
def change_missing_values_to_mean(x_matrix):
    
    x_matrix[x_matrix < 0] = np.nan #Replaces negative values with NaN, so we can take mean
    avg_features = np.nanmean(x_matrix, axis = 0) #Takes mean, ignores NaN values

    for sample in x_matrix:
        
        for i,feature in enumerate(sample):
            if math.isnan(feature):
                sample[i] = avg_features[i]
                
    return x_matrix   

#Adds a binary feature for each feature that has a missing value.
#The feature will contains a 1 if the value is missing, 0 if the feature is present
def add_missing_column_features(x_train, x_validation):
    
    
    count_feats_with_missing_vals = 0
    x_train_missing_values = x_train == -1 #-1 is used if it is a missing value
    x_validation_missing_values = x_validation == -1 #-1 is used if it is a missing value
    for feature_column in np.transpose(x_train_missing_values):
        #If the feature column has a missing value
        if any(feature_column):
            count_feats_with_missing_vals += 1
            
    missing_bools_feats_to_append_to_train = np.zeros(shape = (np.size(x_train, axis = 0), count_feats_with_missing_vals))
    missing_bools_feats_to_append_to_validation = np.zeros(shape = (np.size(x_validation, axis = 0), count_feats_with_missing_vals))
        
    missing_column_index = 0
    
    for i, feature_column_train in enumerate(np.transpose(x_train_missing_values)):
        
        feature_column_validation = np.transpose(x_validation_missing_values)[i]
        
        #If the feature column has a missing value
        if any(feature_column_train): #only train
    
            missing_values_train = np.zeros(len(feature_column_train))
            missing_values_validation = np.zeros(len(feature_column_validation))
            missing_values_train[feature_column_train] = 1
            missing_values_validation[feature_column_validation] = 1
            
            missing_bools_feats_to_append_to_train[:,missing_column_index] = missing_values_train
            missing_bools_feats_to_append_to_validation[:,missing_column_index] = missing_values_validation
            
            
            missing_column_index += 1
            
    return np.append(x_train, missing_bools_feats_to_append_to_train, axis = 1), np.append(x_validation, missing_bools_feats_to_append_to_validation, axis = 1)
    