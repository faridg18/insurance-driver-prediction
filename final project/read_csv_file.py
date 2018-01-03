# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 11:12:25 2017

@author: Gerardo Cervantes
"""
import csv
import numpy as np
import pandas as pd

#Reads CSV files and returns a numpy 2D matrix.
def read_data(x_train_csv, x_test_csv):
    csv_train_items = read_csv(x_train_csv)
    
    x_train = csv_train_items[1:,2:].astype(np.float) #Removes 2 left columns (target and id) and first row (names)
    y_train = csv_train_items[1:,1].astype(np.float) #Gets 2nd column and removes first row 
    
    csv_validation_items = read_csv(x_test_csv)
    
    x_validation = csv_validation_items[0:,1:].astype(np.float)
    validation_ids = csv_validation_items[0:,:1].astype(np.int32).flatten() #Flatten convert matrix of (n,1) to (n,) vector 

    
    return x_train, y_train, x_validation, validation_ids

def read_csv(csv_file_name):
    return pd.read_csv(csv_file_name).as_matrix()


#Exports into a format that Porto Seguro uses for the kaggle competition
def export_csv_file(export_path, val_ids, nn_output):
    
    with open(export_path, "w", newline="\n", encoding="utf-8") as f:
        
        writer = csv.writer(f)
        writer.writerow(["id", "target"])
        for i, val_item in enumerate(nn_output):
            arr = [val_ids[i], val_item[0]]
            writer.writerow(arr)
    