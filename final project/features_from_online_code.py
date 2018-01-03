# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 19:04:31 2017

@author: Jerry C
"""

import pandas as pd
import numpy as np

#Uses feature engineering tool found online in the kaggle discussion page
def features_from_online_code():
    
    
    X_train = pd.read_csv('train.csv')
    X_test = pd.read_csv('test.csv')
    
    x_train, y_train = X_train.iloc[:,2:], X_train.target
    x_validation, test_id = X_test.iloc[:,1:], X_test.id
    
    #OHE / some feature engineering adapted from the1owl kernel at:
    #https://www.kaggle.com/the1owl/forza-baseline/code
    
    #excluded columns based on snowdog's old school nn kernel at:
    #https://www.kaggle.com/snowdog/old-school-nnet
    
    x_train['negative_one_vals'] = np.sum((x_train==-1).values, axis=1)
    x_validation['negative_one_vals'] = np.sum((x_validation==-1).values, axis=1)
    
    to_drop = ['ps_car_11_cat', 'ps_ind_14', 'ps_car_11', 'ps_car_14', 'ps_ind_06_bin', 
               'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 
               'ps_ind_13_bin']
    
    cols_use = [c for c in x_train.columns if (not c.startswith('ps_calc_'))
                 & (not c in to_drop)]
                 
    x_train = x_train[cols_use]
    x_validation = x_validation[cols_use]
    
    one_hot = {c: list(x_train[c].unique()) for c in x_train.columns}
    
    #note that this encodes the negative_one_vals column as well
    for c in one_hot:
        if len(one_hot[c])>2 and len(one_hot[c]) < 105:
            for val in one_hot[c]:
                newcol = c + '_oh_' + str(val)
                x_train[newcol] = (x_train[c].values == val).astype(np.int)
                x_validation[newcol] = (x_validation[c].values == val).astype(np.int)
            x_train.drop(labels=[c], axis=1, inplace=True)
            x_validation.drop(labels=[c], axis=1, inplace=True)
                
    x_train = x_train.replace(-1, np.NaN)  # Get rid of -1 while computing interaction col
    x_validation = x_validation.replace(-1, np.NaN)
    
    x_train['ps_car_13_x_ps_reg_03'] = x_train['ps_car_13'] * x_train['ps_reg_03']
    x_validation['ps_car_13_x_ps_reg_03'] = x_validation['ps_car_13'] * x_validation['ps_reg_03']
    
    x_train = x_train.fillna(-1)
    x_validation = x_validation.fillna(-1)
    
    return x_train.values, y_train.values, x_validation.values, test_id.values