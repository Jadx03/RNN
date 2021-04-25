import numpy as np
from numpy import random
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


# Check if GPU is available
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
else: 
    print('No GPU available, training on CPU; consider making n_epochs very small.')

# train_on_gpu = False

# Open shakespeare text file and read in data as `text`
# with open('tiny-shakespeare.txt', 'r') as f:
#    text = f.read()

text = []
with open("tiny-shakespeare.txt", 'r') as file:
    for line in file:
        for word in line.split():
            text.append(word)
            text.append(" ")
        text.append("\n")


print(len(text))


# Encoding the text and map each character to an integer and vice versa
# 1. intChar, which maps integers to characters
# 2. charInt, which maps characters to integers
chars = tuple(set(text))
intChar = dict(enumerate(chars))
charInt = {ch: ii for ii, ch in intChar.items()}
vocab_size = len(chars)


# Encode the whole text
encoded = np.array([charInt[ch] for ch in text])


# Defining method to encode one hot labels
def one_hot_encode(arr, n_labels):
    
    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot


# Method to make mini-batches for training
def get_batches(arr, batch_size, seq_length):

    
    batch_size_total = batch_size * seq_length
    # Total number of batches we can make
    n_batches = len(arr)//batch_size_total
    
    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))
    
    print("RANGE, ", range(0, arr.shape[1], seq_length))

    # Iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n+seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y


# DEFINING THE ARCHITECTURE
# Define and print the model
n_hidden = 256
n_layers = 2
 

# Declaring the model - LSTM RNN
class CharRNN(nn.Module):
    
    def __init__(self, n_hidden=256, n_layers=2, drop_prob=0.5):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        # Define the LSTM
        self.lstm = nn.LSTM(vocab_size, n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        
        # Define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        # Define the final, fully-connected output layer
        self.fc = nn.Linear(n_hidden, vocab_size)
      
    
    def forward(self, x, hidden):    
        # Get the outputs and the new hidden state from the lstm
        r_output, hidden = self.lstm(x, hidden)
        
        # Pass through a dropout layer
        out = self.dropout(r_output)
        
        # Stack up LSTM outputs using view
        out = out.contiguous().view(-1, self.n_hidden)
        
        # Put x through the fully-connected layer
        out = self.fc(out)
        
        # Teturn the final output and the hidden state
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # Initialized to zero, for hidden state and cell state of LSTM
        # weight = next(self.parameters()).data
       #  hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(), weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(), weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return hidden


# Declare the model
model = CharRNN(n_hidden, n_layers)
print(model)


# Declaring the hyperparameters
batch_size = 100
seq_length = 10
epochs = 10
lr = 0.003
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


# Declaring the train method
def train(model, data, clip=5):

    model.train()   
    if(train_on_gpu):
        model.cuda()
    
    counter = 0
    for e in range(epochs):
        
        # Initialize hidden state
        h = model.init_hidden(batch_size)
        latest_loss = 0
        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1
            print(counter)
            # One-hot encode our data and make them Torch tensors
            x = one_hot_encode(x, vocab_size)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if(train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            h = tuple([each.data for each in h])

            # Zero accumulated gradients
            model.zero_grad()
 
            # Get the output from the model
            output, h = model(inputs, h)
    
            # Calculate the loss and perform backprop
            loss = criterion(output, targets.view(batch_size*seq_length).long())
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            
            latest_loss = loss.item()
            

        print("Epoch", e, "completed. Loss = {:.4f}...".format(latest_loss))              

# Training the model
train(model, encoded)


# Defining a method to generate the next character
def predict(model, char, h=None, top_k=None):

        # Tensor inputs
        x = np.array([[charInt[char]]])
        x = one_hot_encode(x, vocab_size)
        inputs = torch.from_numpy(x)
        
        if(train_on_gpu):
            inputs = inputs.cuda()
        
        # Detach hidden state from history
        h = tuple([each.data for each in h])
        # get the output of the model
        out, h =  model(inputs, h)

        # Get the character probabilities
        p = F.softmax(out, dim=1).data
        if(train_on_gpu):
            p = p.cpu() 
        
        # Get top characters
        if top_k is None:
            top_ch = np.arange(vocab_size)
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()
        
        # Select the likely next character with some element of randomness
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())
        
        # return the encoded value of the predicted char and the hidden state
        return intChar[char], h
        

# Declaring a method to generate new text
def sample(model, size, prime=["QUEEN"], top_k=None):
        
    if(train_on_gpu):
        model.cuda()
    else:
        model.cpu()
    
    # Eval mode
    model.eval() 
    
    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = model.init_hidden(1)
    for ch in prime:
        char, h = predict(model, ch, h, top_k=top_k)

    chars.append(char)
    
    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(model, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)
    

# Generating new text
start = ["QUEEN"]
print(sample(model, 1000, prime=start, top_k=5))
