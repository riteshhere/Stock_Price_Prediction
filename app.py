import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from keras.models import load_model
import streamlit as st


st.title('STOCK PRICE PREDICTION')

user_input = st.text_input('Enter Stock Name:', 'AAPL')
df = pdr.DataReader(user_input, 'yahoo')
df.head()

st.subheader('All stock data till today')
st.write(df.describe())

#Visualization
st.subheader('Closing Price vs TIme Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs TIme Chart with MA100')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(df.Close, 'b')
st.pyplot(fig)

st.subheader('Closing Price vs TIme Chart with MA100 & MA200')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

#splitting data into trainig and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))


#load my model

model = load_model('kears_model2.h5')

past100data = data_training.tail(100)

final_df = past100data.append(data_testing, ignore_index = True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]) :
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])


x_test , y_test = np.array(x_test), np.array(y_test)

#ModelPredition
y_pred = model.predict(x_test)


scale_factor = 1/scaler.scale_
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor



st.subheader('Prediction vs Orignal')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label='Original Values')
plt.plot(y_pred, 'y', label='Predicted Values')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)



