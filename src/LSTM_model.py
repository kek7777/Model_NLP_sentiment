# Import packages
from torch import nn 

# model architecture LSTM

class RNN(nn.Module):
    def __init__(self, vocab_size, output_size, embed_dim, hidden_size=128,  n_layers=2, dropout=0.2):
        super(RNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)                      # embedding layer is useful to map input into vector representation
        
        self.lstm = nn.LSTM(embed_dim, hidden_size, n_layers, dropout=dropout, batch_first=True)    # LSTM layer preserved by PyTorch library
                                                                                                    # batch_first=True - input data format (batch, sequence, features)
        
        # self.dropout = nn.Dropout(0.3)                                              # dropout layer     

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)             # Linear layer for output
        self.sigmoid = nn.Sigmoid()                                # Sigmoid layer cz we will have binary classification


    def forward(self, x):
        x = x.long()                        # changes to type (int64) for embedding
        out = self.embedding(x)             # (batch, seq_len) -> (batch, seq_len, embed_dim)
        out, _ =  self.lstm(out)             # # (batch, seq_len, embed_dim) -> (batch, seq_len, hidden_size)

        out = out[:, -1, :]                 # берет только последний выход последовательности
                                            # Использует только последний выход LSTM - подходит для задач, где важен общий контекст
        out = self.fc1(out)
        out = self.relu(out)
        # apply dropout and fully connected layer
        # out = self.dropout(o)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out