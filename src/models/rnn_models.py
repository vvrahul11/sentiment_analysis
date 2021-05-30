import numpy as np
import pandas as pd

import nltk
import nltk
#nltk.download('punkt')

from nltk import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torchtext
from torchtext.data import Field, BucketIterator, TabularDataset

import torch.optim as optim


def train(model, iterator, optimizer, criterion):
        
        epoch_loss = 0
        epoch_acc = 0        
        model.train()
        
        for batch in iterator:
            
            optimizer.zero_grad()
                    
            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.labels)
            
            rounded_preds = torch.round(torch.sigmoid(predictions))
            correct = (rounded_preds == batch.labels).float() 
            
            acc = correct.sum() / len(correct)
            
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
        return epoch_loss / len(iterator), epoch_acc / len(iterator)


class RNN(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        # convert to embeddings
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        #creates an RNN one word at a time is feeded
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
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


if __name__ == '__main__':
    # parameters
    embedding_dim = 100
    hidden_dim = 256
    output_dim = 1    
    batch_size = 64
    num_epochs = 5
    epoch_loss = 0
    epoch_acc = 0

    
    data = pd.read_csv('data/external/ham-spam/spam.csv', encoding='latin-1')
    data = data.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)
    data = data.rename(index = str, columns = {'v1': 'labels', 'v2': 'text'})

    train, test = train_test_split(data, test_size = 0.2, random_state = 42)
    train.reset_index(drop=True), test.reset_index(drop=True)

    train.to_csv('data/external/ham-spam/train.csv', index=False)
    test.to_csv('data/external/ham-spam/test.csv', index=False)

    TEXT = torchtext.data.Field(tokenize = word_tokenize)
    LABEL = torchtext.data.LabelField(dtype = torch.float)
    datafields = [("labels", LABEL), ("text", TEXT)]

    trn, tst = torchtext.data.TabularDataset.splits(path = 'data/external/ham-spam', 
                                                train = 'train.csv',
                                                test = 'test.csv' ,    
                                                format = 'csv',
                                                skip_header = True,
                                                fields = datafields)

    print(f'Number of training examples: {len(trn)}')
    print(f'Number of testing examples: {len(tst)}')
    print(trn[5].__dict__.keys())
    print(trn[5].text)
    print(trn[5].labels)
    print(print(vars(trn.examples[5])))
    
    for l in range(0,len(trn)):
        print(vars(trn.examples[l]))

    TEXT.build_vocab(trn, max_size = 10500)
    LABEL.build_vocab(trn)
    print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
    print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
    print(TEXT.vocab.freqs.most_common(50))
    print(TEXT.vocab.itos[:10])
    print(LABEL.vocab.stoi)

    train_iterator, test_iterator = torchtext.data.BucketIterator.splits(
                                    (trn, tst),
                                        batch_size = batch_size,
                                        sort_key = lambda x: len(x.text), 
                                        sort_within_batch = False)

    input_dim = len(TEXT.vocab)
    
    model = RNN(input_dim, embedding_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr = 1e-6)
    criterion = nn.BCEWithLogitsLoss()

    def train(model, iterator, optimizer, criterion):
        
        epoch_loss = 0
        epoch_acc = 0        
        model.train()
        
        for batch in iterator:
            
            optimizer.zero_grad()
                    
            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.labels)
            
            rounded_preds = torch.round(torch.sigmoid(predictions))
            correct = (rounded_preds == batch.labels).float() 
            
            acc = correct.sum() / len(correct)
            
            loss.backward()
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)        
        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% ')


    model.eval()
    with torch.no_grad():

        for batch in test_iterator:

            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.labels)

            rounded_preds = torch.round(torch.sigmoid(predictions))
            
            correct = (rounded_preds == batch.labels).float() 
            acc = correct.sum() / len(correct)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    test_loss = epoch_loss / len(test_iterator)
    test_acc  = epoch_acc / len(test_iterator)

    print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')

