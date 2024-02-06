
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf

from keras.models import Sequential,load_model
from keras.layers import LSTM,Dense,Dropout
from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import StandardScaler
from collections import deque
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt

def price_data_processing(stock_data): # define a function to process the stock price data
    pre_days = 10 # predict the tenth day's price

    stock_data['lable']= stock_data['Close'].shift(-pre_days) # this 'lable' is the price after 10days, using realy prices for 10days later if there are value after 10days
    
    # this can used for long short term memory
    # but for the last ten days, no real price for the tenth day, so the value will be none

    scaler = StandardScaler() #for standardizing features , all features have the same scale.
    #print(stock_data.iloc[:,:-1])
    sca_stock = scaler.fit_transform(stock_data.iloc[:,:-1])# all rows. all colums exceppt for the last column. the last column is the predict price, do not need to scale
    #print(type(sca_stock))
    #print(sca_stock[0])
    memory_days = 5 # use five days' prices to predict 

    deq_x = deque(maxlen = memory_days) # create a deque to save the data for five days
    X=[] # X is the input data for the model to predict,every five days' data is an element of X 
    for i in sca_stock:
        deq_x.append(list(i)) #  # i is one day's data, save it in the deque
        #print(deq_x)
        if len(deq_x)==memory_days: # once there are 5 days's data in one deque, put this deque in X
            X.append(list(deq_x)) # one deque is an element in X, each eelement in the X has 5 days' price data 
            #print(X)
        
    X_lately = X[-pre_days:]  # last ten days data . Cause the last ten days had no real the tenth day's price, we slice them out for later predicting 
    
    X=X[:-pre_days]  # all the data except for the last ten days.# so we make sure that all Data in X has a real price for the tenth day
        
    #print(len(X))

    y = stock_data['lable'].values[memory_days-1:-pre_days] # y is the correspinding price for the 10th day,
    #cause we user five day's data to predict the tenth day, so the first 4 days cannot predict future, we cut the first four days out.
    # And the last ten days have no realy price, its value is none, so we cut them out too 
    #print(len(y))
    #print(y)


    X=np.array(X) # transform to the numpy format
    y=np.array(y)
    X_lately=np.array(X_lately)
    #print(X.shape)
    #print(y.shape)
    return X,y,X_lately

