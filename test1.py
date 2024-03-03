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
        output = self.fc(lstmOut)
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


#* time-step [14 days worth of data in one frame]
timestep = 14

#* filled with data from the 14 days [timestep]
xData = []
#* filled with binary values on whether or not the price went up or down[answer-sheet] 
yData = []

for index in range(timestep, len(dataScale) - timestep - 1): 
    xData.append(dataScale[index - timestep: index, 0])
    # yData.append(1 if dataScale['Close'].iloc[index + timestep + 1] > dataScale['Close'].iloc[index + timestep] else 0)
    yData.append(1 if dataScale[index + timestep + 1, 0] > dataScale[index + timestep, 0] else 0)


x, y = torch.tensor(xData, dtype=torch.float32), torch.tensor(yData, dtype=torch.float32)

#* Split the data into training and evaluating sets 
xTrain, yTrain, xEval, yEval = train_test_split(x, y, test_size=0.2, random_state=18)
print(f'This is xTrain shape {xTrain.shape}')
print(f'This is xTrain shape {yTrain.shape}')
print(f'This is xTrain shape {xEval.shape}')
print(f'This is xTrain shape {yEval.shape}')


#* Instantiate the LSTM Model 
inputSize = x.shape[1]
# hidden layer size 
hiddenSize = 200 
# for binary classification, we only want 0 or 1 indicating yes/no up/down
outputSize = 14
model = LSTM(inputSize, hiddenSize, outputSize)

#* Checking for availability of using dedicated GPU or just CPU for computation
component = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(component)
xTrain = xTrain.to(component)
yTrain = yTrain.to(component)

#* Define loss function and optimizer [Binary Cross Entropy Loss :: ADAM optimizer]
criterion = nn.BCELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

#* Ensure that we will always have the same length sequence throughout the training process 
xPad = pad_sequence([xTrain[i:i + timestep] for i in range(0, len(xTrain), timestep)], batch_first=True)
yPad = pad_sequence([yTrain[i:i + timestep] for i in range(0, len(yTrain), timestep)], batch_first=True)


epochNum = 50

for epoch in range(epochNum): 
    # Explicitly set into training mode
    model.train() 
    # clear any previous loss gradient from the previous epoch
    optimizer.zero_grad()
    totLoss = 0 

    for xBatch, yBatch in zip(xPad, yPad):
        # print('****************************')
        # print(f'This is inputs {xBatch}')
        # print(f'This is target {yBatch}')

        # forward pass into the model
        output = model(xBatch)
        # print('---------------------------------')
        # print(f'This is output shape {output.shape}')

        # calculate the loss between predicted guess and expected guess
        loss = criterion(output.squeeze(), yBatch)

        totLoss += loss.item()

        # backward pass [calculate the loss gradient]
        loss.backward()

        # optimizer [adjust weights and biases (parameters)]
        optimizer.step()

    avgLoss = totLoss / (len(xTrain) // timestep) 
    print(f'Epoch [{epoch + 1}/{epochNum}], Loss: {avgLoss}')


#* Evaluation
xEval = xEval.to(component)
yEval = yEval.to(component)


xEvalPad = pad_sequence([xEval[i:i + timestep] for i in range(0, len(xEval), timestep)], batch_first=True)
yEvalPad = pad_sequence([yEval[i:i + timestep] for i in range(0, len(yEval), timestep)], batch_first=True)

# set to evaluation mode
model.eval()

# disable loss gradient
with torch.no_grad(): 
    totLoss = 0
    correct = 0 
    totSample = 0 

    for xBatch, yBatch, in zip(xEvalPad, yEvalPad): 
        # pass the data forward [add a batch dimension here]
        output = model(xBatch.unsqueeze(0))

        # calculate the loss from the predicted to the expected answer
        loss = criterion(output.squeeze(), yBatch)
        totLoss += loss.item()

        # round the labels 
        predictedLabel = torch.round(output)

        correct += (predictedLabel == yBatch).sum().item()
        totSample += len(yBatch)

    evalLoss = totLoss / (len(xEval) // timestep)
    print(f'This is correct {correct} and this is totSample {totSample}') 
    accuracy = correct / totSample
    print(f'Evaluation Accuracy: {accuracy * 100}%, Evaluation Loss: {evalLoss}')