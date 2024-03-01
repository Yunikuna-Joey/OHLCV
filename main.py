import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn 
import torch.optim as optim
from sklearn.model_selection import train_test_split
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
    
# Load and preprocess OHLCV data
df = pd.read_csv('./data/AAPL.csv')
# .head() provides the first few rows of data
# print(df.head())

# normalize the values using the formula z standardized score = (original value - mean) / standard deviation of data 
normdf = (df[['Open', 'High', 'Low', 'Close', 'Volume']] - df[['Open', 'High', 'Low', 'Close', 'Volume']].mean()) / df[['Open', 'High', 'Low', 'Close', 'Volume']].std()
# print(normdf.head())

#* define time step (what is equal to 1 time step) in this case we are doing 14 days (2 weeks) worth of data is == 1 time step
timeStep = 14

# Create input sequences and target labels
X = []      # filled with data from 10 consecutive days
y = []      # filled with binary 0 or 1 to indicate whether the price went up

for i in range(len(normdf) - timeStep - 1):
    # [ [14 days worth of close#1], 
    #   [14 days worth of close#2], 
    #   [14 days worth of close#3],
    # ]
    X.append(normdf.iloc[i:i+timeStep].values)

    #* compares the closing values from the 11th day to the 10th day, if higher, then 1 else 0
    y.append(1 if normdf['Close'].iloc[i + timeStep + 1] > normdf['Close'].iloc[i + timeStep] else 0)

# converts to tensor-type variables to be used to train model with 
X = torch.tensor(X, dtype=torch.float32)            # normalized df[open, high, low, close, volume]
y = torch.tensor(y, dtype=torch.float32)

# Split the data into training and testing sets
Xtrain, Xeval, ytrain, yeval = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the model
input_size = X.shape[2]             #* X.shape [(batch_size) - number of sequences, (sequence_len) - how much in 1 individual sequence, (num_features) - columns of data ] 
# how many nodes are in the hidden player
hidden_size = 100
# what the output value will be which is 1 scalar value
output_size = 1
# create the model using our class defintion 
model = LSTM(input_size, hidden_size, output_size)

# checking for GPU availability
component = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(component)
X_train = Xtrain.to(component)
y_train = ytrain.to(component)

# Define loss function and optimizer
criterion = nn.BCELoss()            # *Binary Cross Entropy loss measures the difference between predicted probabilities (scores) and the actual binary label. 
                                    # *Punishes the model more heavily when it makes confident incorrect predictions 
                                    # *Like a validator after a guess was made

#* optimizer = adapts the learning rate for each parameter individually 
optimizer = optim.Adam(model.parameters(), lr=0.001)        #* lr = learning rate... lower the value = slower, but stable training... [smaller jumps]
                                                            #* it adjusts how much the model learns from each mistake and how quickly it adjusts its guesses


#* new variables here [ensure that we will always have the same length sequence]
padX = pad_sequence([X_train[i:i+timeStep] for i in range(0, len(X_train), timeStep)], batch_first=True)
padY = pad_sequence([y_train[i:i+timeStep] for i in range(0, len(y_train), timeStep)], batch_first=True)

#* the amount of times the entire dataset is used 
num_epochs = 20
for epoch in range(num_epochs):
    # set to train mode 
    model.train()
    # clear the previous gradient losses
    optimizer.zero_grad()
    total_loss = 0

    for batch_X, batch_y in zip(padX, padY): 
        # forward pass of the normalized data
        output = model(batch_X)
        # calculate the loss between the model predicted output ['output'] against the y_train list [already consists of the true binary answers] [answer sheet]
        loss = criterion(output.squeeze(), batch_y)                  #* squeeze seems to be maintenance [when we expect scalar values, sometimes training will output list of lists [[1], [0], [0], etc]]           
        total_loss += loss.item()                                    #* squeeze will remove the list of lists to ensure we get scalar values [example]
        # backward pass [calculate the loss gradient]
        loss.backward()                                             #* loss gradient ?= parameter values == weights [multiply] and biases [add]
                                                                    #* loss gradient represents how much the loss function changes as each parameter of the model changes
        # Updates the parameter values(?) based on the computed loss gradients using the optimization algorithm defined by optimizer
        optimizer.step()                                            #* the optimization algorithm uses the loss gradients to make its decision on how to optimize its decision making
                                                                    #* the algorithm uses the gradients internally 

        
    # Print average loss for the epoch
    average_loss = total_loss / (len(X_train) // timeStep)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}')

Xeval = Xeval.to(component)
yeval = yeval.to(component)

testX = pad_sequence([Xeval[i:i+timeStep] for i in range(0, len(Xeval), timeStep)], batch_first=True)
testY = pad_sequence([yeval[i:i+timeStep] for i in range(0, len(yeval), timeStep)], batch_first=True)

# Evaluate the model
model.eval()                                            #* explicity set the mode to evaluation

# 'with' keyword temp. disables loss gradient calculation
with torch.no_grad():
    test_loss = 0
    correct = 0 
    totSample = 0 

    for batch_X, batch_y in zip(testX, testY):
        # passes the test data forward 
        test_output = model(batch_X)

        # maintenance portion... ensure that predicted output is in the format needed [removes unnecessary dimensions from the output] and calculates the loss between prediction and true value
        batch_loss = criterion(test_output.squeeze(), batch_y)
        test_loss += batch_loss.item()

        # rounds the predicted labels to the nearest 0 or 1 
        predicted_labels = torch.round(test_output)

        # calculate accuracy comparing the predicted labels to the answer key where they are == then divide by sample size
        correct += (predicted_labels == batch_y).sum().item()
        totSample += len(batch_y)
    
    # prints loss and accuracy
    average_test_loss = test_loss / (len(Xeval) // timeStep)
    accuracy = correct / totSample
    print(f'Test Loss: {average_test_loss}, Test Accuracy: {accuracy}%')
