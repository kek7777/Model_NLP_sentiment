# Import packages
import torch
import os
from torch import nn 
from torch import optim 
from torch.utils.data import DataLoader



# model architecture LSTM

class LSTM(nn.Module):
    def __init__(self, vocab_size, output_size, embed_dim, hidden_size=128,  n_layers=2, dropout=0.2):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)                      # embedding layer is useful to map input into vector representation

        
        self.rnn = nn.LSTM(embed_dim, hidden_size, n_layers, dropout=dropout, batch_first=True)    # LSTM layer preserved by PyTorch library

        
        # self.dropout = nn.Dropout(0.3)                                              # dropout layer     

        self.fc1 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_size, output_size)                               # Linear layer for output

        
        self.sigmoid = nn.Sigmoid()                                                 # Sigmoid layer cz we will have binary classification

    def forward(self, x):
        
        # convert feature to long
        # x = x.long()

        # map input to vector
        out = self.embedding(x)

        # pass forward to lstm
        out, _ =  self.rnn(out)

        # get last sequence output
        out = out[:, -1, :]

        out = self.fc1(out)
        out = self.relu(out)

        # apply dropout and fully connected layer
        # out = self.dropout(o)
        out = self.fc2(out)

        # sigmoid
        out = self.sigmoid(out)

        return out