import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

#Step 0: Split corpus into "segments". Here, I've used sentences, but you could use anything else (lines, for example). 

# sentences = ["Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much.", "They were the last people you’d expect to be involved in anything strange or mysterious, because they just didn’t hold with such nonsense.", "Mr. Dursley was the director of a firm called Grunnings, which made drills.", "He was a big, beefy man with hardly any neck, although he did have a very large mustache.", "Mrs. Dursley was thin and blonde and had nearly twice the usual amount of neck, which came in very useful as she spent so much of her time craning over garden fences, spying on the neighbors.", "The Dursleys had a small son called Dudley and in their opinion there was no finer boy anywhere."]

# Words
# opening the text file
word_count  = 0
words = []
with open("tiny-shakespeare.txt",'r') as file:
   
    # reading each line    
    for line in file:
   
        # reading each word        
        for word in line.split():
            
            word_count += 1
            # displaying the words           
            words.append(word)


print(len(words))
set_of_chars = set(words)

intChar = dict(enumerate(set_of_chars))
charInt = {character: index for index, character in intChar.items()}

all_characters = charInt
print(len(all_characters))
print(words.index("and"))
cc = input()
'''
all_characters = "".join(set_of_chars)
all_characters = sorted(all_characters)
all_characters = "".join(all_characters)
'''
n_characters = len(all_characters)

#First step in pre-processing: Convert everything, every character in your vocabulary into a one-hot
#This isn't the only way to do things, you could, for example, use an embedding layer. 

#Vocabulary setup
#Extracts all characters from sentences
characters = set(''.join(sentences))

#I like to set up 2 dictionaries, one to convert characters to integers, and one to convert back
intChar = dict(enumerate(characters))
charInt = {character: index for index, character in intChar.items()}

#Still setting up one-hots, but we're going to offset things now. 

#Example: Sequence = Mr. and Mrs. | Input = Mr. and Mrs | Target = r. and Mrs. (These characters are now paired into input,target pairs. 
#NOT THE ONLY WAY TO DO THINGS: You could end every sequence with a special token signifying the end of a sequence (ex: <eos>).
input_sequence = []
target_sequence = []

#Set these up by removing last character in the input, and removing the first character in the target
for i in range(len(sentences)):
    input_sequence.append(sentences[i][:-1])
    target_sequence.append(sentences[i][1:])
#print(target_sequence)
    
#Replace all characters with integer values
for i in range(len(sentences)):
    input_sequence[i] = [charInt[character] for character in input_sequence[i]]
    target_sequence[i] = [charInt[character] for character in target_sequence[i]]
    
#print(target_sequence)

vocab_size = len(charInt)

def create_one_hot(sequence, vocab_size):
    #Define a matrix of size vocab_size containing all 0s
    #Dimensions: Batch Size x Sequence Length x Vocab Size
    #Have to do this even if your batch size is 1
    encoding = np.zeros((1,len(sequence),vocab_size), dtype=np.float32)
    for i in range(len(sequence)):
        encoding[0,i,sequence[i]] = 1
    #Return a sequence of one-hot encodings    
    return encoding
    
#print(create_one_hot(input_sequence[0], vocab_size))

#Next, let's define the RNN model
class RNNModel(nn.Module):
    #Num_layers refers to the number of layers in your hidden state
    #batch_first indicates that the first index is the batch index
    
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        #Default batch index is 1 and not 0. 
        #(sequence,batch,word) - batch_first = false
        #(batch,sequence,word) - batch_first = true
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        #This is to make sure that our output dimension is correct
        self.fc = nn.Linear(hidden_size, output_size)
        
        
        
    
    def forward(self, x):
        hidden_state = self.init_hidden()
        
        output,hidden_state = self.rnn(x, hidden_state)
        
        #Use this to deal with the extra dimension from having a batch
        output = output.contiguous().view(-1,self.hidden_size)
        
        output = self.fc(output)
        
        return output, hidden_state
    
    def init_hidden(self):
        #Remember, (row, BATCH, column)
        hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        return hidden
        
model = RNNModel(vocab_size, vocab_size, 100, 1)

#Set up loss and optimizers
#Cross Entropy Loss will automatically perform softmax on the outputs for you
loss = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters())

#TRAINING LOOP!!!!
for epoch in range(150):
    for i in range(len(input_sequence)):
        optimizer.zero_grad()
        #Create our input as a tensor
        x = torch.from_numpy(create_one_hot(input_sequence[i], vocab_size))
        
        #Create target
        #Cross entropy loss uses integers as targets. No need for one hots
        y = torch.Tensor(target_sequence[i])
        
        
        output,hidden = model(x)
        
        lossValue = loss(output, y.view(-1).long())
        lossValue.backward()
        optimizer.step()
        
        print("Loss: {:.4f}".format(lossValue.item()))
        
        
def predict(model, character):
    characterInput = np.array([charInt[c] for c in character])
    characterInput = create_one_hot(characterInput, vocab_size)
    characterInput = torch.from_numpy(characterInput)
    out,hidden = model(characterInput)
    
    prob = nn.functional.softmax(out[-1],dim=0).data
    character_index = torch.max(prob, dim=0)[1].item()
    
    return intChar[character_index],hidden
    
def sample(model, out_len, start='The'):
    characters = [ch for ch in start]
    currentSize = out_len - len(characters)
    for i in range(currentSize):
        character, hidden_state = predict(model, characters)
        characters.append(character)
        
    return ''.join(characters)
    
print(sample(model, 50))