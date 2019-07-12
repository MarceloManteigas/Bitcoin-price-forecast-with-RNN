# -*- coding: utf-8 -*-
#Part 1 - Data preprocessing

#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import training set
dataset_train = pd.read_csv('Coinbase_BTCUSD_d_train.csv')
training_set = dataset_train.iloc[:,2:3].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)
#creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []

for i in range(240,1641):
    X_train.append(training_set_scaled[i-240:i,0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train),np.array(y_train)
#reshaping the data
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))


#Part 2 - Building Rnn

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initializing the RNN
regressor = Sequential()
#Adding the first LSTM layer and some Droupouts
regressor.add(LSTM(units = 150, return_sequences = True, input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.4)) 
#Adding the second LSTM layer and some Droupouts
regressor.add(LSTM(units = 150, return_sequences = True))
regressor.add(Dropout(0.4))
#Adding the third LSTM layer and some Droupouts
regressor.add(LSTM(units = 150, return_sequences = True))
regressor.add(Dropout(0.4))
#Adding the third LSTM layer and some Droupouts
regressor.add(LSTM(units = 150, return_sequences = True))
regressor.add(Dropout(0.4))
#Adding the third LSTM layer and some Droupouts
regressor.add(LSTM(units = 150, return_sequences = True))
regressor.add(Dropout(0.4))


#Adding the forth LSTM layer and some Droupouts
regressor.add(LSTM(units = 150))
regressor.add(Dropout(0.4))
#Adding the output layer
regressor.add(Dense(units = 1))
#Compiling the RNN
regressor.compile(optimizer = 'adam', loss='mean_squared_error')
#fitting the RNN to the training set
regressor.fit(X_train,y_train, epochs = 100, batch_size = 10)

#Part 3 - Predictions and visualization
'''
I'm putting the data on the opposite direction
'''
#Getting the real stock price of 2017
dataset_test = pd.read_csv('Coinbase_BTCUSD_d_test.csv')
real_stock_price = dataset_test.iloc[:,2:3].values

#Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis = 0) #veritcal concatenation axis = 0
inputs = dataset_total[len(dataset_total)-len(dataset_test)-240:].values
inputs = inputs.reshape(-1,1) #reshape to match the same dimensions of the RNN
inputs = sc.transform(inputs) #not fit because you dont want to change the object
X_test = []
for i in range(240,269):
    X_test.append(inputs[i-240:i,0])
X_test= np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualizing the results
plt.plot(real_stock_price,color='red', label = 'Real Bitcoin Stock Price')
plt.plot(predicted_stock_price ,color='blue', label = 'Predicted Bitcoin Stock Price')
plt.title('Bitoin Strock Price Predictions')
plt.xlabel('Time')
plt.ylabel('Bitcoin Stock Price')
plt.legend()
plt.show()

# Part 4 - Evaluating the RNN

import math
from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print("RMSE =", rmse)

# Save model

regressor.save('cryptomodel.h5')