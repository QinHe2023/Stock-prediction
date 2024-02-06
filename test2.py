
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
from data_process import price_data_processing



stock_stymbol = 'TSLA'
stock_info = yf.Ticker(stock_stymbol)

stock_data = stock_info.history(period="1d",start='2000-01-01',end='2023-12-31')

stock_data.drop(['Dividends','Stock Splits'],axis=1, inplace=True)
#print(stock_data)
stock_data.dropna(inplace=True)
stock_data.sort_index(inplace=True)
#print(stock_data)

X,y,X_lately = price_data_processing(stock_data)

print(len(X))
print(len(y))