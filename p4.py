'''
Name: Jainam Dhruva
Project 4: Shaksphere text using RNN
Date: Apr 21, 2021
University of Kentucky
'''
import random
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


# Reading the text file and converting it to a one big string
shakesphere_input = open("tiny-shakespeare.txt", "r")
shakesphere_text = shakesphere_input.read()
shakesphere_input.close()

# Setting up the Vocabulary
# Extracts all characters from sentences
characters = set(shakesphere_text)
vocab_size = len(characters)
batch_size = 1


# Making two dictionaries for one hot encoding and further use
intChar = dict(enumerate(characters)) # { int : char} <- an integer for each char
charInt = {character: index for index, character in intChar.items()} # { char : int} <- a char for each int

# --------------------------------------------------- ONE-HOT ENCODING -----------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------

input_sequence = []
target_sequence = []

# We use a sliding window technique to generate strings of length 50
# Input will be offset by one at the end
# Target will be offset by one at the star
str_len = 300
for i in range(0, len(shakesphere_text)-1, str_len):
    if (i + str_len < len(shakesphere_text) - 1):
        input_sequence.append(shakesphere_text[i : i + str_len])
        target_sequence.append(shakesphere_text[i+1 : i + (str_len+1)])
    else:
        input_sequence.append(shakesphere_text[i : len(shakesphere_text)-1 ])
        target_sequence.append(shakesphere_text[i+1 : len(shakesphere_text) ])

total_inputs = len(input_sequence)
for i in range(0, total_inputs):
    input_sequence[i] = [charInt[character] for character in input_sequence[i]]
    target_sequence[i] = [charInt[character] for character in target_sequence[i]]


for i in range(0, total_inputs):
   print( len(input_sequence[i]), len(target_sequence[i]))


def create_one_hot(sequence, vocab_size):
    
    #Define a matrix of size vocab_size containing all 0s
    #Dimensions: Batch Size x Sequence Length x Vocab Size
    #Have to do this even if your batch size is 1
    encoding = np.zeros((1,len(sequence),vocab_size), dtype=np.float32)
    for i in range(len(sequence)):
        encoding[0,i,sequence[i]] = 1
    
    #Return a sequence of one-hot encodings    
    return encoding    





# ------------------------------------------------------- RNN MODEL --------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------

class RNNModel(nn.Module):
    
    #Num_layers refers to the number of layers in your hidden state
    def __init__(self, input_size, output_size, hidden_size, num_layers, drop_prob):
        
        super(RNNModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        #batch_first indicates that the first index is the batch index
        #Default batch index is 1 and not 0. 
        #(sequence,batch,word) - batch_first = false
        #(batch,sequence,word) - batch_first = true
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=drop_prob, batch_first=True)
        
        # Dropout Layer
        self.dropout = nn.Dropout(drop_prob)

        #This is to make sure that our output dimension is correct
        self.fc = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, x, hidden_state):
        # hidden_state = self.init_hidden()
        
        output,hidden_state = self.lstm(x, hidden_state)
        
        #pass through a dropout layer
        output = self.dropout(output)

        #Use this to deal with the extra dimension from having a batch
        output = output.contiguous().view(-1,self.hidden_size)
        
        output = self.fc(output)
        
        return output, hidden_state
    
    def init_hidden(self, batch_size):
        #Remember, (row, BATCH, column)
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                      weight.new(self.num_layers, batch_size, self.hidden_size).zero_())
        return hidden
        
model = RNNModel(vocab_size, vocab_size, 20, 2, 0.3)


#Set up loss and optimizers
#Cross Entropy Loss will automatically perform softmax on the outputs for you
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.5)

#TRAINING LOOP!!!!
for epoch in range(1):
    print("Epoch: ", epoch)
    h = model.init_hidden(batch_size) 
    print(len(input_sequence))
    for i in range(len(input_sequence)):
        h = tuple([each.data for each in h])
        optimizer.zero_grad()
        #Create our input as a tensor
        x = torch.from_numpy(create_one_hot(input_sequence[i], vocab_size))
        
        #Create target
        #Cross entropy loss uses integers as targets. No need for one hots
        y = torch.Tensor(target_sequence[i])
        
        output,hidden = model(x, h)
        
        lossValue = loss(output, y.view(-1).long())
        lossValue.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)   
        optimizer.step()
        
        print("i = ", i , "Loss: ", lossValue.item() )
    
        
        
def predict(model, character, h):
    characterInput = np.array([charInt[c] for c in character])
    characterInput = create_one_hot(characterInput, vocab_size)
    characterInput = torch.from_numpy(characterInput)

    h = tuple([each.data for each in h])
    out,hidden = model(characterInput, h)
    
    prob = nn.functional.softmax(out[-1],dim=0).data
    character_index = torch.max(prob, dim=0)[1].item()
    
    return intChar[character_index],hidden
    
def sample(model, out_len, start='QUEEN'):
    characters = [ch for ch in start]
    currentSize = out_len - len(characters)
    for i in range(currentSize):
        character, hidden_state = predict(model, characters, h)
        characters.append(character)
        
    return ''.join(characters)
    
print(sample(model, 100))







print("Hello World")

