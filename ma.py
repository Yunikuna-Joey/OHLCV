import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn 
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

df = pd.read_csv('./data/AAPL.csv')
# Attempt to calculate moving averages 
windowSize = 10

# creates a new column in df variable and it will be the moving average of the close price
df['Close_MA'] = df['Close'].rolling(window=windowSize).mean()

df['Close_MA'].fillna(method='bfill', inplace=True)
 
print(df.head())

# Preprocess the data
xMA = []
yMA = []

# fill in the expected output here
for i in range(len(df) - 1): 
    yMA.append(1 if df['Close'].iloc[i + 1] > df['Close'].iloc[i] else 0) 


# after filling in expected data, normalize/standardized them
scaler = StandardScaler()
xMAscaled = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume', 'Close_MA']].values)

# convert into tensor objects here 
xMA = torch.tensor(xMAscaled[:-1], dtype=torch.float32)
yMA = torch.tensor(yMA, dtype=torch.float32)

# print(f'This is the length of X {len(xMA)} and Y {len(yMA)}')
# print(f'This is xMA.shape {xMA.shape}')

print(f'This is xMA {xMA}')
print(f'This is yMA {yMA}')

# split the data into training sets 
xTrain, yTrain, xEval, yEval = train_test_split(xMA, yMA, test_size=0.2, random_state=42)

# instantiate the model 
input_size = xMA.shape[1]

# hidden layer
hidden_size = 100

# output 
output_size = 1 

model = LSTM(input_size, hidden_size, output_size)

timeStep = 7

# checking for GPU availability
component = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(component)
xData = xTrain.to(component)
yData = yTrain.to(component)

# pad to match lengths of input and output 
padX = pad_sequence([xData[i:i+timeStep] for i in range(0, len(xData), timeStep)], batch_first=True)
padY = pad_sequence([yData[i:i+timeStep] for i in range(0, len(yData), timeStep)], batch_first=True)
print(f'pad x size {padX.size()}')
print(f'pad y size {padY.size()}')

# define loss function and optimizer 
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochNum = 25

for epoch in range(epochNum): 
    model.train() 
    optimizer.zero_grad()
    totLoss = 0

    for xBatch, yBatch in zip(padX, padY): 
        # print(f'This is xBatch {xBatch}')
        # print(f'This is yBatch {yBatch}')
        # forward pass the normalized data
        output = model(xBatch)
        print(f'This is output {output}')
        print(f'This is output shape {output.shape}')
        # calculate the loss between model prediction and real prediction
        loss = criterion(output, yBatch)
        totLoss += loss.item()
        # backward pass [calculate the loss gradient]
        loss.backward()

        optimizer.step()

    avgLoss = totLoss / len(xTrain) // timeStep
    print(f'Epoch [{epoch+1}/{epochNum}], Loss: {avgLoss}')


# Evaluation stage
# xEval = xEval.to(component)
# yEval = yEval.to(component)
