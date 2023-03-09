import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
import pickle

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math

start = '2010-01-01'
end = '2023-03-01'
st.title('Stock Market Prediction')

user_input = st.text_input("Enter Stock Ticker", 'GOOG')
df = yf.download(user_input, start , end)


#describing Data

st.subheader("Data from 2010 - 2023 ")
st.write(df.describe())

#visualization 
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.grid()
st.pyplot(fig)

st.subheader('Volume Trends')
fig, ax = plt.subplots(figsize=(20,8))
ax.plot(df['Volume'])
ax.xaxis.set_major_locator(plt.MaxNLocator(15))
ax.set_xlabel('Date', fontsize='11')
ax.set_ylabel('Volumes', fontsize='11')
plt.title('Volume Trends', fontsize='20')
plt.grid()
st.pyplot(fig)

st.subheader('Market Cap')
df['Market Cap'] = df['Open']*df['Volume']
fig, ax = plt.subplots(figsize=(20,8))
ax.plot(df['Market Cap'], color='orange')
ax.xaxis.set_major_locator(plt.MaxNLocator(15))
ax.set_xlabel('Date', fontsize='11')
ax.set_ylabel('Market Cap', fontsize='11')
plt.title('Market Cap')
plt.grid()
st.pyplot(fig)

st.subheader('Volatility')
df['vol'] = (df['Close']/df['Close'].shift(1)) - 1
fig, ax = plt.subplots(figsize=(20,8))
ax.plot(df['vol'], color='purple')
ax.xaxis.set_major_locator(plt.MaxNLocator(15))
plt.title('Volatility')
plt.grid()
plt.show()
st.pyplot(fig)

st.subheader('Cumulative Return')
fig, ax = plt.subplots(figsize=(20,8))
df['Cumulative Return'] = (1 + df['vol']).cumprod()
ax.plot(df['Cumulative Return'], color='green')
ax.xaxis.set_major_locator(plt.MaxNLocator(15))
ax.set_xlabel('Date', fontsize='11')
ax.set_ylabel('Cumulative Return', fontsize='11')
plt.title('Cumulative Return')
plt.grid()
plt.show()
st.pyplot(fig)

# st.subheader('Closing Price vs Time Chart with 100MA')
# ma100 = df.Close.rolling(100).mean()
# fig=plt.figure(figsize = (12,6))
# plt.plot(ma100)
# plt.plot(df.Close)
# plt.grid()
# st.pyplot(fig)

# st.subheader('Closing Price vs Time Chart with 100MA And 200MA')
# ma100 = df.Close.rolling(100).mean()
# ma200 = df.Close.rolling(200).mean()
# fig=plt.figure(figsize = (12,6))
# plt.plot(ma100)
# plt.plot(ma200)
# plt.plot(df.Close)
# plt.grid()
# st.pyplot(fig)

model = pickle.load(open("Stock_market.pkl",'rb'))

# past_100_days = data_training.tail(100)
# final_df = past_100_days.append(data_testing , ignore_index=True)
# input_data = scaler.fit_transform(final_df)

# x_test = []
# y_test = []

# for i in range(100, input_data.shape[0]):
#     x_test.append(input_data[i-100:i])
#     y_test.append(input_data[i,0])
    
# x_test,y_test = np.array(x_test) , np.array(y_test)


scaler = MinMaxScaler(feature_range=(0,1))
new_df= df.filter(['Close'])
last_60_days = new_df[-60:].values
last_60_days_scaled = scaler.fit_transform(last_60_days)
X_test = []
# y_test = last_60_days
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
pred_price= model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)


# data  = df.filter(['Close'])

# train = data[:-60]
# valid = data[-60:]
# valid['Predictions'] = pred_price
# fig, ax = plt.subplots(figsize=(20,8))
# ax.plot(valid[['Close','Predictions']],linewidth=3.5)
# plt.title('Model',fontsize=18)
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price' ,fontsize=18)
# plt.legend(['Valid','Predictions'], loc='upper left')
# st.pyplot(fig)

# scaler = scaler.scale_

# rmse = np.sqrt(np.mean(pred_price - y_test)**2)
# st.write(rmse)
# scale_factor = 1 / scaler[0]
# y_predicted = y_predicted*scale_factor
# y_test = y_test * scale_factor

# st.subheader('Predictions vs Original')
# fig2 = plt.figure(figsize=(12,6))
# plt.plot(y_test , 'b',label = 'Original price')
# plt.plot(pred_price, 'r', label = 'Predicted Price')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# st.pyplot(fig2)
st.write( f"The Predicted Price for next day  is {pred_price}" )

last_60_days = new_df[-10:].values
st.subheader('Prediction')
fig2 = plt.figure(figsize=(12,6))
plt.plot(last_60_days, 'b',label = 'Original price')
plt.plot(pred_price, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
