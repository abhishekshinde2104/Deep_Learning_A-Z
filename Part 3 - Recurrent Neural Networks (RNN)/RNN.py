#DATA PREPROCESSING

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing Training sets
"""We are going to train our training set only and our model wont have any idea on the test set
"""
dataset_train=pd.read_csv("Google_Stock_Price_Train.csv")
training_set=dataset_train.iloc[:,1:2].values

#Feature Scaling(standardisation and normalisation)
#mostly in rnn whenever there is a sigmoid activation function in ouput we should apply normalisation
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))#feature range ==> all prices between 0 and 1
training_set_scaled=sc.fit_transform(training_set)
#fit means it will just get min and max stock price 
#transform means it will calculate for each stock price the scaled stock price acc to the formula

"""next step --> CREATING A SPECIFIC DATA STRUCTURE 
.)most important step of data preprocessing for rnn
.)we are going tto create a data structure specifying what the rnn will need to remember when 
predicting the next stock price and this is called THE NUMBER OF TIME STEPS.
impt as wrong number of time steps might lead to overfitting"""

#Creating a data structure with 60 timesteps and 1 output
"""60 timesteps means that at each timestep T is going to look at 60 stock prices before time T
(i.e stock prices of 60 days before time T) and based on this its going to predict the next output
1 output is at time T+1"""
#x_train will contain the info abt 60 timesteps before time T
#y_train will contain the output at T+1
X_train = []
y_train = []
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
X_train,y_train=np.array(X_train),np.array(y_train)
#rnn is memorizing whats happening in before 60 timesteps to predict at T+1

#Reshaping the data

#i.e adding more  dimensionality to the previous data structure
#this dimension is going to be the unit i.e the number of predictors we can use to predict what we want
#write now we 1 indicator i.e Open Google Stock Price
#we will now add more indicators 

#we will do it only for X_train and this will also make it compatible as the input shape for the rnn
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
"""Input to RNN"""
#we will make it 3D as for rnn we require 3D tensor
#input shape==(batch_size,timesteps,input_size)
#batch_size->rows
#timesteps->columns
#input_size->no of indicators


#Part 2 - Building the RNN

#importing libraries
from keras.models import Sequential#nn object representing sequence of layers
from keras.layers import Dense##output layer
from keras.layers import LSTM#lstm layer
from keras.layers import Dropout#dropout regularization

#Initialise RNN
regressor=Sequential()#this time we are predicting continuous value

#Adding first LSTM layer and dropout regularization
#droupout regularization is applied to avoid overfitting 

regressor.add(LSTM(units=50 , return_sequences=True , input_shape=(X_train.shape[1],1)))
#units->no of cells you want to have in lstm layer/neurons
#return_sequences=True->we set it true as a stacked lstm as you are going to add more lstm layers
#input_shape->last step of data preprocessing (3D) and  we only specify the timesteps and indicators

regressor.add(Dropout(rate=0.2))
#rate -> rate at you want to drop the neurons i.e u want to ignore the neurons 

#Adding second LSTM layer and dropout regularization
regressor.add(LSTM(units=50 , return_sequences=True))
#as this is the 2nd layer bcoz this we dont specify the input_shape
regressor.add(Dropout(rate=0.2))

#Adding third LSTM layer and dropout regularization
regressor.add(LSTM(units=50 , return_sequences=True))
regressor.add(Dropout(rate=0.2))

#Adding fourth LSTM layer and dropout regularization
regressor.add(LSTM(units=50))#default return_sequnces=False as  we dont add any new lstm layer
regressor.add(Dropout(rate=0.2))

#Adding Output layer
regressor.add(Dense(units=1))
#units->no of neurons in the output layer

#Compiling the RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')
#rmsprop->usually recommended for the rnn
#as this is a regression prblem loss will be mean squared error

#Fitting the RNN to Training set
regressor.fit(X_train,y_train,epochs=100,batch_size=32)


#PART 3 : Making Predictions and Visualisation the results

#Getting the real stock price of 2017
dataset_test=pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test.iloc[:,1:2].values
 
#Getting predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values

inputs=inputs.reshape(-1,1)

inputs=sc.transform(inputs)#ssame scaling as that of training set

X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)


X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

predicted_stock_price = regressor.predict(X_test)

predicted_stock_price=sc.inverse_transform(predicted_stock_price)

#Visualising Results

plt.plot(real_stock_price,color='red',label='Real Google Stock Price')
plt.plot(predicted_stock_price,color='blue',label='Predicted Google Stock Price')
plt.title("Google Stock Price Predictions")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()


















