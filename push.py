import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn 
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.nn.utils.rnn import pad_sequence

# base class for neural network module (PyTorch)
class LSTM(nn.Module): 
    # input layer, hidden layer, and output layer of a Nueral Network Model 
    def __init__(self, inputSize, hiddenSize, outputSize): 
        # constructor of nn.Module by using super
        super(LSTM, self).__init__()
        # size of hidden layer 
        self.hiddenSize = hiddenSize
        # initialize an lstm layer with input features and hidden units
        self.lstm = nn.LSTM(inputSize, hiddenSize)
        # initialize a Fully Connected (linear) layer that maps the output of LSTM layer => output layer
        self.fc = nn.Linear(hiddenSize, outputSize)
        # activation function applied to the output of the fc layer
        self.sigmoid = nn.Sigmoid()
    
    #* Might want to experiment with sigmoid , ReLu, and/or Softplus functions and see results
    # this is how data will be passed (forwarded) throughout the neural network layer
    def forward(self, input):
        # the output of lstm contains the features for each time step, '_' contains the hidden state and cell state (not used) 
        lstmOut, _ = self.lstm(input)
        # applies the fc layer to the output of lstm layer @ the last time step
        output = self.fc(lstmOut[-1])
        # apply the sigmoid function to squash the values to the range [0, 1]
        output = self.sigmoid(output)

        # represents the predicted values 
        return output

# Compact apple historical data into a panda data frame
df = pd.read_csv('./data/AAPL.csv')

#* set the rule of the scaler to hold values between 0 and 1 [effectively normalizing the data]
scaler = MinMaxScaler(feature_range=(0, 1))

#* .values() converts the values into a NumPy array... 
#* reshape(num. of rows is auto determined based on len of original array, states there should only be ONE column)
#* essentially, this makes it so that every value has its own row
dataScale = scaler.fit_transform(df['Close'].values.reshape(-1, 1))     #* numpy.ndarray type

print('The data scale is type', type(dataScale) )

#* time-step [14 days worth of data in one frame]
timestep = 14

#* filled with data from the 14 days [timestep]
xData = []
#* filled with binary values on whether or not the price went up or down[answer-sheet] 
yData = []

for index in range(timestep, len(dataScale)): 
    xData.append(dataScale[index - timestep: index, 0])
    # yData.append(1 if dataScale['Close'].iloc[index + timestep + 1] > dataScale['Close'].iloc[index + timestep] else 0)
    yData.append(1 if dataScale)


xTrain, yTrain = torch.tensor(xData, dtype=torch.float32), torch.tensor(yData, dtype=torch.float32)


print('This is dataScale', dataScale)
print('This is xTrain', xTrain)
print('This is yTrain', yTrain)
print('---------------------------------')
print('This is xTrain shape', xTrain.shape)
print('This is yTrain shape', yTrain.shape)

