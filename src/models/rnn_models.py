
import torch.nn as nn

class RNN(nn.module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        # convert to embeddings
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        #creates an RNN one word at a time is feeded
        self.rnn = rnn.RNN(embedding_dim, hidden_dim)
        #fully connected layer gives the outut
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        # one batch is feed. list of indexed of the word, gets the embedding
        # output: sentence length, batch size , hidden dim
        embedded = self.embedding(text)
        # one part is feeded
        # output: 1, batchsize, hidden size
        output, hidden = self.rnn(embedded)
        # just get the last hidden state for each sentence
        hidden_1D = hidden.squeeze(0)
        # double check if it is the last hidden state
        assert torch.equal(output[-1, :, :], hidden_1D)

        return self.fc(hidden_1D)


class LSTM(nn.Module):
    
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim)
  
        self.rnn = nn.LSTM(embedding_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        
        embedded = self.embedding(text)
        
        output, (hidden, _) = self.rnn(embedded)
        
        hidden_1D = hidden.squeeze(0)
        
        assert torch.equal(output[-1, :, :], hidden_1D)
        
        return self.fc(hidden_1D)

class LSTM_DROPOUT(nn.Module):
    
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim)
  
        self.rnn = nn.LSTM(embedding_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, text):
        
        embedded = self.embedding(text)
        
        embedded_dropout = self.dropout(embedded)
        
        output, (hidden, _) = self.rnn(embedded_dropout)
        
        hidden_1D = hidden.squeeze(0)
        
        assert torch.equal(output[-1, :, :], hidden_1D)
        
        return self.fc(hidden_1D)