import random
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import random
import unidecode

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
# Get characters from string.printable
all_characters = string.printable
n_characters = len(all_characters)
print(type(all_characters))
'''
'''
# Read large text file (Note can be any text file: not limited to just names)
file = unidecode.unidecode(open("tiny-shakespeare.txt").read())
'''

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


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(input_size, hidden_size) 
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        out = self.embed(x)
        out, (hidden, cell) = self.lstm(out.unsqueeze(1), (hidden, cell))
        out = self.fc(out.reshape(out.shape[0], -1))
        return out, (hidden, cell)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden, cell


class Generator:
    def __init__(self):
        self.chunk_len = 250
        self.num_epochs = 5000
        self.batch_size = 1
        self.print_every = 50
        self.hidden_size = 256
        self.num_layers = 2
        self.lr = 0.003


    def char_tensor(self, string_list):
        tensor = torch.zeros(len(string_list)).long()
        for c in range(len(string_list)):
            tensor[c] = all_characters[string_list[c]]
        return tensor

    def get_random_batch(self):
        start_idx = random.randint(0, word_count - self.chunk_len)
        end_idx = start_idx + (self.chunk_len + 1)
        text_str = words[start_idx:end_idx]
        text_input = torch.zeros(self.batch_size, self.chunk_len)
        text_target = torch.zeros(self.batch_size, self.chunk_len)

        for i in range(self.batch_size):
            text_input[i, :] = self.char_tensor(text_str[:-1])
            text_target[i, :] = self.char_tensor(text_str[1:])

        return text_input.long(), text_target.long()

    def generate(self, initial_str="QUEEN", predict_len=100, temperature=0.85):
        hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)
        initial_input = self.char_tensor(initial_str)
        predicted = initial_str

        for p in range(len(initial_str) - 1):
            output, (hidden, cell) = self.rnn(initial_input[p].view(1).to(device), hidden, cell )

        last_char = initial_input[-1]

        for p in range(predict_len):
            output, (hidden, cell) = self.rnn(last_char.view(1).to(device), hidden, cell)

            output_dist = output.data.view(-1).div(temperature).exp()
            top_char = torch.multinomial(output_dist, 1)[0]
            predicted_char = all_characters[top_char]
            
            predicted += predicted_char
            last_char = self.char_tensor(predicted_char)

        return predicted

    # input_size, hidden_size, num_layers, output_size
    def train(self):
        self.rnn = RNN( n_characters, self.hidden_size, self.num_layers, n_characters).to(device)

        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        print("=> Starting training")

        for epoch in range(1, self.num_epochs + 1):
            inp, target = self.get_random_batch()
            hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)

            self.rnn.zero_grad()
            loss = 0
            inp = inp.to(device)
            target = target.to(device)

            for c in range(self.chunk_len):
                output, (hidden, cell) = self.rnn(inp[:, c], hidden, cell)
                loss += criterion(output, target[:, c])

            loss.backward()
            optimizer.step()
            loss = loss.item() / self.chunk_len

            if epoch % self.print_every == 0:
                print(f"Loss: {loss}")
                print(self.generate())




gennames = Generator()
gennames.train()