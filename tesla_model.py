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



stock_stymbol = 'TSLA'
stock_info = yf.Ticker(stock_stymbol)


stock_data = stock_info.history(period="1d",start='2000-01-01',end='2023-12-28')

stock_data.drop(['Dividends','Stock Splits'],axis=1, inplace=True)
#print(stock_data)
stock_data.dropna(inplace=True)
stock_data.sort_index(inplace=True)
#print(stock_data)

X,y,X_lately = price_data_processing(stock_data)
#print('XXXXXXXXXXXXXXXXXXXXXXXXxx')
#print(X_lately)#
X_train,X_test,y_train,y_test = train_test_split(X,y,shuffle=False,test_size=0.1)
# print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaa')
# print(X_test)
# print(X_test.shape)
# print('VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV')
# print(X_train)
# print(X_train.shape)

print("5555555555555555555555")
print(X.shape)
print(X.shape[1:])


model = Sequential()
model.add(LSTM(10,input_shape = X.shape[1:],activation='relu',return_sequences=True)) # 神经网络添加第一层
model.add(Dropout(0.1)) # 防止overfit

model.add(LSTM(10,activation='relu',return_sequences=True)) # 神经网络添加第二层
model.add(Dropout(0.1)) # 防止overfit

model.add(LSTM(10,activation='relu')) # 神经网络添加第三层
model.add(Dropout(0.1)) # 防止overfit

model.add(Dense(10,activation='relu')) # 全连接层
model.add(Dropout(0.1)) # 防止overfit

model.add(Dense(1)) #输出层 1

model.compile(optimizer='adam',
            loss='mse',#回归方差 平均的  在scilearn regression  B 站视频第4个结尾部分
            metrics=['mape']) #编译

model.fit(X_train,y_train,batch_size = 32,epochs=10,validation_data = (X_test,y_test))
model.summary()
model.evaluate(X_test,y_test)
predict_price = model.predict(X_test) # 为什么用test 数组来测而不是train？

stock_time = stock_data.index[-len(y_test):]

plt.plot(stock_time,y_test,color='red',label='price')
plt.plot(stock_time,predict_price,color='green',label='price')
plt.show()
