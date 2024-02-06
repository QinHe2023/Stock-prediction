from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf

from keras.models import Sequential,load_model
from keras.layers import LSTM,Dense,Dropout
from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import StandardScaler
from collections import deque
from sklearn.model_selection import train_test_split # scikit learn documentation 
import matplotlib.pyplot as plt
from data_process import price_data_processing
from data_process_optimize import price_data_processing_optimize

def predict_apple_one_day():
    stock_stymbol = 'AAPL'
    stock_info = yf.Ticker(stock_stymbol)
    today_date = datetime.today().strftime('%Y-%m-%d')

    stock_data = stock_info.history(period="1d",start='2000-01-01',end=today_date)
    print(stock_data)

    stock_data.drop(['Dividends','Stock Splits'],axis=1, inplace=True)
    #print(stock_data)
    stock_data.dropna(inplace=True)
    stock_data.sort_index(inplace=True)
    #print(stock_data)

    X,y,X_lately = price_data_processing_optimize(stock_data,5,1)

    #print(X_lately)
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,shuffle=False,test_size=0.1) 
    # use scikit-learn funciton train_test_split  to split a dataset into training and testing sets. 
    

    # print(X_test)
    # print(X_test.shape)
    # print(X_train)
    # print(X_train.shape)

    #print("5555555555555555555555")
    #print(X.shape)
    #print(X.shape[1:])

    model = Sequential()
    model.add(LSTM(10,input_shape = X.shape[1:],activation='relu',return_sequences=True)) # add the first LSTM layer
    model.add(Dropout(0.1)) # prevent overfit

    model.add(LSTM(10,activation='relu',return_sequences=True)) # add the second LSTM layer
    model.add(Dropout(0.1)) # prevent overfit

    model.add(LSTM(10,activation='relu')) # add the third LSTM layer
    model.add(Dropout(0.1)) # prevent overfit

    #model.add(Dense(10,activation='relu')) # add the fourth Dense layer
    #model.add(Dropout(0.1)) # prevent overfit

    model.add(Dense(1)) # add the output layer 

    model.compile(optimizer='adam',
                loss='mse',#回归方差 平均的  在scilearn regression  
                metrics=['mape']) #编译

    model.fit(X_train,y_train,batch_size = 32,epochs=10,validation_data = (X_test,y_test))
    model.summary()
    model.evaluate(X_test,y_test)
    predict_price = model.predict(X_test)

    stock_time = stock_data.index[-len(y_test):]

    #plt.plot(stock_time,y_test,color='red',label='price')
    #plt.plot(stock_time,predict_price,color='green',label='price')
    #plt.show()
    
    return y_test,predict_price,stock_time,model,X_lately

