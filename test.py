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
from sklearn.model_selection import train_test_split # 可以查看scikit learn documentation 
import matplotlib.pyplot as plt

import streamlit as st

from model_google import predict_google


# stock_stymbol = 'GOOG'
# stock_info = yf.Ticker(stock_stymbol)

# stock_data = stock_info.history(period="1d",start='2000-01-01',end='2023-12-01')

# stock_data.drop(['Dividends','Stock Splits'],axis=1, inplace=True)
# print(stock_data)
# stock_data.dropna(inplace=True)
# pre_days = 10
# stock_data['lable']= stock_data['Close'].shift(-pre_days)
# print(stock_data.head(23))




# scaler = StandardScaler()
# print(stock_data.iloc[:,:-1])# all rows. all colums exceppt for the last colum
# sca_stock = scaler.fit_transform(stock_data.iloc[:,:-1])
# print(type(sca_stock))
# print(sca_stock[0])
# memory_days = 5
# deq_x = deque(maxlen = memory_days)



# predict_data = yf.Ticker('GOOG') #get the data of the selected stock
# predict_data_period = predict_data.history(period="1d",start='2023-12-1',end='2023-12-28') 

# #print(predict_data_period["Close"])
# #print(type(predict_data_period["Close"]))
# st.line_chart(predict_data_period["Close"])

# ##print(predict_data_period)
# #print(type(predict_data_period))

# predict_data_period.drop(['Dividends','Stock Splits','Volume'],axis=1, inplace=True)
# st.line_chart(predict_data_period)




# real_price,predict_price,stock_time,model = predict_google()
# print(type(real_price))
# print(real_price)
# print(type(predict_price))
# print(predict_price)
# print(type(stock_time))
# print(stock_time)

# print(predict_price.shape)
# print(real_price.shape)
#predict_price_reshape = predict_price.reshape(-1)
# print(predict_price_reshape.shape)

# df = pd.DataFrame({
#      'stock_time': stock_time,
#      'predict_price':predict_price_reshape,
#      'real_price': real_price
 
#  })

# df.set_index('stock_time', inplace=True)

# print(df)


# st.line_chart(df)


stock_stymbol = 'GOOG'
stock_info = yf.Ticker(stock_stymbol)



x_data_period = stock_info.history(period="1d",start='2023-12-20',end='2023-12-28') 

x_data_period.drop(['Dividends','Stock Splits'],axis=1, inplace=True)
# print(x_data_period)

scaler = StandardScaler()

sca_x = scaler.fit_transform(x_data_period)
# print(sca_x)
sca_x_reshape = sca_x.reshape(1,5,5)
# print(sca_x_reshape)
# print(sca_x_reshape.shape)




real_price,predict_price,stock_time,model,last_ten_days = predict_google()
print('test-----------------')
#print(last_ten_days.shape)

predict_price_reshape = predict_price.reshape(-1)
y_pre = model.predict(sca_x_reshape) # 只有5天数据，scaler 时不对，要用原始数据中最后10天的 X_lately
#print(y_pre)
pre = model.predict(last_ten_days )
pre_reshape = pre.reshape(-1)
all_predict_price = np.hstack((predict_price_reshape, pre_reshape))
#print(all_predict_price)

pre_days = np.arange(1, 11)

real_price_for_last_10_days = np.zeros(10)
real_price_new = np.hstack((real_price, real_price_for_last_10_days))

stock_time_new = np.hstack((stock_time, pre_days))

#print(pre)
df = pd.DataFrame({
    'stock_time': stock_time,
    'predict_price':predict_price_reshape,
    'real_price': real_price

})
df.set_index('stock_time', inplace=True)
st.line_chart(df)


st.line_chart(pre_reshape)




# deq_x = deque(maxlen = 5)
# X=[]
# for i in sca_x:
#     print(i)
#     deq_x.append(list(i))
#     #print(deq_x)
#     if len(deq_x)==memory_days:
#         X.append(list(deq_x))
#         #print(X)
    
# X_lately = X[-pre_days:] 
# X=X[:-pre_days]    
            


