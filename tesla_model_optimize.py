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
from sklearn.model_selection import train_test_split # scikit learn documentation 
import matplotlib.pyplot as plt
from data_process import price_data_processing
from data_process_optimize import price_data_processing_optimize


stock_stymbol = 'TSLA'
stock_info = yf.Ticker(stock_stymbol)

stock_data = stock_info.history(period="1d",start='2000-01-01',end='2023-12-28')

stock_data.drop(['Dividends','Stock Splits'],axis=1, inplace=True)
#print(stock_data)
stock_data.dropna(inplace=True)
stock_data.sort_index(inplace=True)
#print(stock_data)



#print(X_lately)


# print(X_test)
# print(X_test.shape)
# print(X_train)
# print(X_train.shape)

# print("5555555555555555555555")
# print(X.shape)
# print(X.shape[1:])

mem_days = [5,10,15]
pre_days = 10
lstm_layers = [1,2,3]
dense_layers = [1,2,3]
units = [16,32]

for the_mem_days in mem_days:
    for the_lstm_layers in lstm_layers:
        for the_dense_layers in dense_layers:
            for the_units in units:
                filepath =  './models/tesla/{val_mape:.2f}_{epoch:02d}_'+f'mem_{the_mem_days}_lstm_{the_lstm_layers}_dense_{the_dense_layers}_units_{the_units}'
                #print(filepath)
                checkpoint = ModelCheckpoint(
                    filepath=filepath,
                    save_weights_only=True,
                    monitor='val_mape',
                    mode='min',# the samller the better 
                    save_best_only=True
                )
                X,y,X_lately = price_data_processing_optimize(stock_data,the_mem_days,pre_days)
                X_train,X_test,y_train,y_test = train_test_split(X,y,shuffle=False,test_size=0.1) 
                # use scikit-learn funciton train_test_split  to split a dataset into training and testing sets. 
                model = Sequential()
                model.add(LSTM(the_units,input_shape = X.shape[1:],activation='relu',return_sequences=True)) # add the first LSTM layer
                model.add(Dropout(0.1)) # prevent overfit
                
                for i in range(the_lstm_layers):

                    model.add(LSTM(the_units,activation='relu',return_sequences=True)) # add the second LSTM layer
                    model.add(Dropout(0.1)) # prevent overfit

                model.add(LSTM(the_units,activation='relu')) # add the third LSTM layer
                model.add(Dropout(0.1)) # prevent overfit
                
                for i in range(the_dense_layers):

                    model.add(Dense(the_units,activation='relu')) # add the fourth Dense layer
                    model.add(Dropout(0.1)) # prevent overfit

                model.add(Dense(1)) # add the output layer 

                model.compile(optimizer='adam',
                            loss='mse',#回归方差 平均的  在scilearn regression  
                            metrics=['mape']) #编译

                model.fit(X_train,y_train,batch_size = 32,epochs=10,validation_data = (X_test,y_test),callbacks=[checkpoint])





model.summary()
model.evaluate(X_test,y_test)
predict_price = model.predict(X_test)

stock_time = stock_data.index[-len(y_test):]

plt.plot(stock_time,y_test,color='red',label='price')
plt.plot(stock_time,predict_price,color='green',label='price')
plt.show()
    


