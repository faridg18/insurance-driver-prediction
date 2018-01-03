# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:37:54 2017

@author: Gerardo Cervantes
"""

import numpy as np
#Used to plots graphs in python
import matplotlib.pyplot as plt

#Keras Library for neural networks
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import regularizers, optimizers, initializers

from read_csv_file import read_data, export_csv_file
from roc_auc_callback import roc_auc_callback
from features_from_online_code import features_from_online_code
from preprocess_data import preprocess
from plot_results import plot_results

from sklearn.model_selection import *

#Kwargs, can be given optional arguments: x_test and y_test. The extra data it takes in is used as validation data to see how well the model does at predicting the given data
#Returns history of NN
def trainNeuralNetwork(neuralNetworkModel, batchSize, nEpochs, x_train, y_train, **kwargs):
    
    if (('x_test' in kwargs) & ('y_test' in kwargs)):
        x_test = kwargs['x_test']
        y_test = kwargs['y_test']
        
        gini_callback = roc_auc_callback(training_data=(x_train, y_train),validation_data=(x_test, y_test))
        
        neuralNetworkModel.fit(x=x_train, y=y_train, batch_size=batchSize, epochs=nEpochs, verbose=2, callbacks=[gini_callback], validation_data= (x_test,y_test) )
        
        return gini_callback
    else:
        neuralNetworkModel.fit(x=x_train, y=y_train, batch_size=batchSize, epochs=nEpochs, verbose=2, callbacks=None)
    
#Gives the NN an architecture
def addNeuralNetworkLayers(model, numberInputNodes, numberOutputNodes):
    #Input layer
    model.add(Dense(numberInputNodes, input_shape=(numberInputNodes,), activation = 'relu'))
    model.add(Dense(80, activation='relu', kernel_initializer='normal'))

    model.add(Dense(70, activation='relu', kernel_initializer='normal'))
    model.add(Dense(60, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001)))
   # model.add(Dense(55, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.0005)))
    model.add(Dense(50, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.0001)))
    #model.add(Dropout(0.5))
    #model.add(Dense(45, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.00005)))
    model.add(Dense(40, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.0001)))
   # model.add(Dense(35, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.000005)))
    model.add(Dense(30, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dense(25, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.0000025)))

    #model.add(Dense(20, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.0001)))
    #Output layer
    model.add(Dense(numberOutputNodes, activation='sigmoid', kernel_initializer='normal'))
    '''
    model.add(Dense(512, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(60, activation='relu', kernel_initializer='normal', kernel_regularizer=regularizers.l2(0.001)))
    '''
    #model.add(Dense(numberOutputNodes, activation='sigmoid', kernel_initializer='normal'))
    return model

#high level function - Creates NN from x_train, y_train.
#Creates csv from the x_validation and validation_ids used for kaggle competition
def puertoPredictions(x_train, y_train, x_validation, validation_ids):
    
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size= .15, random_state=42) #Splits to 10% test data
    
    model = Sequential()
    addNeuralNetworkLayers(model, len(x_train[0]), 1) 
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.001), metrics = ["binary_accuracy"])
    
    nEpochs = 60
    batchSize = 512
    
    print('Training!')
    
    #trainNeuralNetwork(model, batchSize, nEpochs, x_train, y_train)
    roc_auc = trainNeuralNetwork(model, batchSize, nEpochs, x_train, y_train, x_test = x_test, y_test = y_test)
    plot_results(roc_auc.gini, roc_auc.gini_val)
    nn_output = model.predict(x_validation)
    export_csv_file("output.csv", validation_ids, nn_output)


use_features_from_online_code = True

if use_features_from_online_code:
    x_train, y_train, x_validation, validation_ids = features_from_online_code();
else:
    x_train, y_train, x_validation, validation_ids = read_data('train.csv', 'test.csv')
    x_train, y_train, x_validation = preprocess(x_train, y_train, x_validation)
    
puertoPredictions(x_train, y_train, x_validation, validation_ids)