# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 08:53:24 2017

@author: Jerry C
"""

#Used to plots graphs in python
import matplotlib.pyplot as plt



def plot_results(gini, gini_val):
    plt.plot(gini)
    plt.plot(gini_val)
    
    plt.title('Model Gini coefficient')
    plt.ylabel('Gini coefficient')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    
    plt.show()