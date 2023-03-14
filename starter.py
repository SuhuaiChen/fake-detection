import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import tqdm
import time
import numpy as np
import sys
import argparse
import os
import torchmetrics
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryConfusionMatrix
from sklearn.metrics import classification_report, precision_recall_fscore_support
import pickle
import csv

def decode(vocab,corpus):

    text = ''
    for i in range(len(corpus)):
        wID = corpus[i]
        text = text + vocab[wID] + ' '
    return(text)

def encode(words,text,model_type):
    corpus = []
    tokens = text.split(' ')
    if 'LSTM' in model_type:
        tokens = text.split()
    for t in tokens:
        try:
            wID = words[t][0]
        except:
            wID = words['<unk>'][0]
        corpus.append(wID)
    return(corpus)

def read_encode(file_name,vocab,words,corpus,threshold,model_type):
    
    wID = len(vocab)
    
    if threshold > -1:
        with open(file_name,'rt', encoding='utf-8') as f:
            for line in f:
                line = line.replace('\n','')
                line = line.replace('\t', ' ')
                tokens = line.split(' ')
                if 'LSTM' in model_type:
                    tokens = line.split()
                for t in tokens:
                    try:
                        elem = words[t]
                    except:
                        elem = [wID,0]
                        vocab.append(t)
                        wID = wID + 1
                    elem[1] = elem[1] + 1
                    words[t] = elem

        temp = words
        words = {}
        vocab = []
        wID = 0
        words['<unk>'] = [wID,100]
        vocab.append('<unk>')
        for t in temp:
            if temp[t][1] >= threshold:
                vocab.append(t)
                wID = wID + 1
                words[t] = [wID,temp[t][1]]
            
                
    with open(file_name,'rt', encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n','')
            line = line.replace('\t', ' ')
            tokens = line.split(' ')
            if 'LSTM' in model_type:
                tokens = line.split()
            for t in tokens:
                try:
                    wID = words[t][0]
                except:
                    wID = words['<unk>'][0]
                corpus.append(wID)
                
    return [vocab,words,corpus]


class FFNN(nn.Module):
    def __init__(self, vocab, words,d_model, d_hidden, window_size, dropout):
        super().__init__() 
    
        self.vocab = vocab
        self.words = words
        self.vocab_size = len(self.vocab)
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.dropout = nn.Dropout(p=dropout)
        self.embeds = nn.Embedding(self.vocab_size,d_model)
        self.hidden = nn.Linear(d_model * window_size, d_hidden)
        self.output = nn.Linear(d_hidden, 1)
        self.sigmoid = nn.Sigmoid()
        self.window_size = window_size

    def forward(self, src):
        embeds = self.embeds(src)
        embedded = self.dropout(embeds)
        embedded = embedded.view(-1, self.window_size * self.d_model)
        hidden = self.hidden(embedded)
        output = self.output(hidden)
        output = self.sigmoid(output)
        return output
    
class LSTM(nn.Module):
    def __init__(self,vocab,words,d_model,d_hidden,n_layers,dropout_rate):
        super().__init__()
        self.vocab = vocab
        self.words = words
        self.vocab_size = len(self.vocab)
        self.n_layers = n_layers
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.embeds = nn.Embedding(self.vocab_size,d_model)
        self.lstm = nn.LSTM(d_model, d_hidden, num_layers=n_layers, 
                    dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.logprobs = nn.Linear(d_hidden, self.vocab_size)
        assert d_model == d_hidden, 'cannot tie, check dims'
        self.embeds.weight = self.logprobs.weight
        
    def forward(self,src,h):
        embeds = self.embeds(src)
        out, h = self.lstm(self.dropout(embeds), h)
        preds = self.logprobs(out)
        return [preds,h]
    
    def init_weights(self):
        pass

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.n_layers, batch_size, self.d_hidden).to(device)
        cell = torch.zeros(self.n_layers, batch_size, self.d_hidden).to(device)
        return hidden, cell
          
    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return [hidden, cell] 
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("Device available for running: ")
    print(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d_model', type=int, default=100)
    parser.add_argument('-d_hidden', type=int, default=100)
    parser.add_argument('-n_layers', type=int, default=2)
    parser.add_argument('-batch_size', type=int, default=20)
    parser.add_argument('-seq_len', type=int, default=30)
    parser.add_argument('-printevery', type=int, default=5000)
    parser.add_argument('-window', type=int, default=3)
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-dropout', type=int, default=0.35)
    parser.add_argument('-clip', type=int, default=2.0)
    parser.add_argument('-model', type=str,default='LSTM')
    parser.add_argument('-savename', type=str,default='model')
    parser.add_argument('-loadname', type=str, default='model')
    parser.add_argument('-trainname', type=str,default='wiki.train.txt')
    parser.add_argument('-validname', type=str,default='wiki.valid.txt')
    parser.add_argument('-testname', type=str,default='wiki.test.txt')
    
    params = parser.parse_args()    

    # for testing
    params.model = 'FFNN'
    params.epochs = 10
    params.batch_size = 32
    params.window = 10
    params.savename = 'ffnnModel'
    params.loadname = 'ffnnModel'
    params.trainname = 'mix.train.txt'
    params.validname = 'mix.valid.txt'
    params.testname = 'mix.test.txt'

    
    torch.manual_seed(0)

    # ------------------ read data and encode ------------------
    # vocab: list of words
    # words: dictionary of words and their ids and counts
    # train: the corpus in terms of word ids
    [vocab,words,train] = read_encode('./data/' + params.trainname,[],{},[],3, params.model)
    print('vocab: %d train: %d' % (len(vocab),len(train)))
    [vocab,words,test] = read_encode('./data/' + params.testname,vocab,words,[],-1, params.model)
    print('vocab: %d test: %d' % (len(vocab),len(test)))
    params.vocab_size = len(vocab)
    [vocab,words,valid] = read_encode('./data/' + params.validname,vocab,words,[],-1, params.model)
    print('vocab: %d test: %d' % (len(vocab),len(valid)))

    FAKE_TOKEN = "[FAKE]"
    REAL_TOKEN = "[REAL]"
    END_TOKEN = "<end_bio>"


    # x: list of b  io texts
    # y: [REAL] (1) or [FAKE] (0)
    def FFNN_getInputAndTarget(train):
        input = []
        temp = []
        for token in train:
            temp.append(token)
            if token == words[FAKE_TOKEN][0] or token == words[REAL_TOKEN][0]:
                input.append(temp)
                temp = []
        for i in range(len(input)):
            if input[i][-1] == words[FAKE_TOKEN][0]:
                input[i] = [input[i][:-1], 0]
            else:
                input[i] = [input[i][:-1], 1]
        x = [i[0] for i in input]
        y = [i[1] for i in input]
        return x, y
    
    def FFNN_getInput(train):
        input = []
        temp = []
        for token in train:
            temp.append(token)
            if token == words[END_TOKEN][0]:
                input.append(temp)
                temp = []
        return input

    # split the input data with given target into windows of window_size
    def FFNN_split_windows(window_size, inputs, target):
        windows = []
        labels = []
        for j in range(len(inputs)):
            data = inputs[j]
            for i in range(len(data) - window_size + 1):
                window = data[i:i + window_size]
                windows.append(window)
                label = target[j]  # Assign binary label based on task
                labels.append(label)
        return windows, labels

    # split the data into windows of window_size
    def FFNN_split_window(window_size, data):
        windows = []
        for i in range(len(data) - window_size + 1):
            window = data[i:i + window_size]
            windows.append(window)
        return windows
    
    def LSTM_split_sequences(data, sequence_length, step):
        sequences = []
        for i in range(0, len(data) - sequence_length + step, step):
            sequence = data[i:i + sequence_length]
            if len(sequence) < sequence_length:
                sequence = np.concatenate([sequence, np.zeros((sequence_length - len(sequence),))])
            sequences.append(torch.Tensor(sequence))
        return sequences
    
    def LSTM_train(model, dataloader, optimizer, criterion, batch_size, seq, device):
        training_loss = 0
        model.train()
        h = model.init_hidden(batch_size, device)
        
        for idx, batch in enumerate(tqdm(dataloader, total=len(dataloader), position=0, leave=True)):
            # if (idx+1) % 1000 == 0:
            #   print("Batch completed: " + str(idx+1))
            optimizer.zero_grad()
            h = model.detach_hidden(h)

            input, labels = batch[:, :-1], batch[:, 1:]
            input, labels = input.to(device).to(torch.long), labels.to(device).to(torch.long)
            preds, h = model(input, h)         
            preds = preds.reshape(batch_size * seq, -1)   
            labels = labels.reshape(-1)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
        return training_loss / len(dataloader)
    
    def LSTM_evaluate(model, dataload, criterion, batch_size, seq, device):
        epoch_loss = 0
        model.eval()
        h = model.init_hidden(batch_size, device)

        with torch.no_grad():
            for idx, data in enumerate(tqdm(dataload, total=len(dataload), position=0, leave=True)):
                h = model.detach_hidden(h)
                src, labels = data[:,:-1], data[:,1:]
                src, labels = src.to(device).to(torch.long), labels.to(device).to(torch.long)
                preds, h = model(src, h)
                preds = preds.reshape(batch_size * seq, -1)
                labels = labels.reshape(-1)
                loss = criterion(preds, labels)
                epoch_loss += loss.item()
        return epoch_loss / len(dataload)

    def LSTM_generate(data):
        model.eval()
        hidden = model.init_hidden(1, device)
        predict = []
        for i, seq in enumerate(tqdm(data, total=len(data))):
            with torch.no_grad():
                src = torch.Tensor(seq).to(device).to(torch.long).unsqueeze(dim=0)
                prediction, hidden = model(src, hidden)
            if words[END_TOKEN][0] in seq:
                probs = torch.softmax(prediction[:, seq.tolist().index(words[FAKE_TOKEN][0])], dim=-1).squeeze()
                if probs[words[FAKE_TOKEN][0]] > probs[words[REAL_TOKEN][0]]:
                    predict.append(0)
                else:
                    predict.append(1)
        return predict
    
    def LSTM_preprocess(train):
        input = []
        temp = []
        for token in train:
            temp.append(token)
            if token == words[FAKE_TOKEN][0] or token == words[REAL_TOKEN][0]:
                input.append(temp)
                temp = []
        for i in range(len(input)):
            if input[i][-1] == words[FAKE_TOKEN][0]:
                input[i] = [input[i][3:len(input[i])-1], 0]
            else:
                input[i] = [input[i][3:len(input[i])-1], 1]
        train_input = [i[0] for i in input]
        train_target = [i[1] for i in input]
        return train_input, train_target


    if params.model == 'FFNN':
#          {add code to instantiate the model, train for K epochs and save model to disk}
        print("setup Training FFNN model...")

        # get x_train, y_train, x_valid, y_valid, x_test, y_test
        # split bios into windows of window_size with label of [REAL] or [FAKE]
        train_input, train_target = FFNN_getInputAndTarget(train)
        valid_input, valid_target = FFNN_getInputAndTarget(valid)
        train_input = [torch.Tensor(i) for i in train_input]
        valid_input = [torch.Tensor(i) for i in valid_input]
        train_windows, train_labels = FFNN_split_windows(params.window, train_input, train_target)
        valid_windows, valid_labels = FFNN_split_windows(params.window, valid_input, valid_target)


        # initialize model and hyperparameters
        model = FFNN(vocab, words, params.d_model, params.d_hidden, params.window, params.dropout)
        model.to(device)
        model.train()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=params.lr)

        # Create a DataLoader for parallel batch training
        train_dataset = [(train_windows[i], train_labels[i]) for i in range(len(train_windows))]
        train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=2)
        valid_dataset = [(valid_windows[i], valid_labels[i]) for i in range(len(valid_windows))]
        valid_loader = DataLoader(valid_dataset, batch_size=params.batch_size, shuffle=True, num_workers=2)

        losses = []
        valid_losses = []
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)
        print("begin Training...")
        # train + validation for k epochs
        for k in range(params.epochs):
            print("epoch: ", k+1)
            train_loss = 0
            valid_loss = 0
            for i, (batch_windows, batch_labels) in enumerate(tqdm(train_loader, total=len(train_loader))):
                optimizer.zero_grad()
                batch_outputs = model(batch_windows.to(torch.int64).to(device))
                batch_labels = batch_labels.to(device).to(torch.float).unsqueeze(1)
                loss = criterion(batch_outputs, batch_labels)

                # L2 regularization
                # _lambda = 0.0001
                # l2_reg = torch.tensor(0.)
                # for param in model.parameters():
                #     l2_reg += torch.norm(param)**2
                # loss += 0.5 * _lambda * l2_reg

                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            with torch.no_grad():
                for i, (batch_windows, batch_labels) in enumerate(tqdm(valid_loader, total=len(valid_loader))):
                    inputs = batch_windows.to(torch.int64).to(device)
                    labels = batch_labels.to(device).to(torch.float).unsqueeze(1)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item()


                inputs = torch.Tensor(torch.stack(valid_windows)).to(torch.int64).to(device)
                outputs = model(inputs)


            # validation = criterion(outputs, torch.Tensor(valid_labels).to(device).to(torch.float).unsqueeze(1)).tolist()
            # valid_losses.append(validation)
            lr_scheduler.step(valid_loss)
            losses.append(train_loss/len(train_loader))
            valid_losses.append(valid_loss/len(valid_loader))
            print("average training loss: ", train_loss/len(train_loader))
            print("average validation loss: ", valid_loss/len(valid_loader))

        # save data to disk
        torch.save(model, './' + params.savename + '.pt') 
        
    if params.model == 'LSTM':
#          {add code to instantiate the model, train for K epochs and save model to disk}

        print("Setup Training LSTM model...")

        train_sq = LSTM_split_sequences(train, params.window + 1, params.window + 1)
        valid_sq = LSTM_split_sequences(valid, params.window + 1, params.window + 1)
        
        dataloader = DataLoader(train_sq, batch_size=params.batch_size, num_workers = 2,shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_sq, batch_size=params.batch_size, num_workers = 2,shuffle=True, drop_last=True)
        
        model = LSTM(vocab, words, params.d_model, params.d_hidden, params.n_layers, params.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
        criterion = nn.CrossEntropyLoss()

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)
        train_losses = []
        valid_losses = []
        print("begin Training...")

        for epoch in range(params.epochs):
            print('epoch: ', epoch + 1)
            train_loss = LSTM_train(model, dataloader, optimizer, criterion, 
                        params.batch_size, params.window, device)
            valid_loss = LSTM_evaluate(model, valid_loader, criterion,  params.batch_size, 
                        params.window, device)
            
            lr_scheduler.step(valid_loss)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

        torch.save(model, './'+ params.savename + '.pt') 

    if params.model == 'FFNN_CLASSIFY':
#          {add code to instantiate the model, recall model parameters and perform/learn classification}

        print("Testing FFNN model...")

        model = torch.load('./'+ params.loadname + '.pt')
        test_input, test_target = FFNN_getInputAndTarget(test)

        test_input = [torch.Tensor(i) for i in test_input]
        model.eval()
        predictions = []
        print("begin testing...")
        for bio in test_input:
            windows = FFNN_split_window(params.window, bio) # split each testing bio into windows
            if len(bio) < params.window:
                predictions.append(torch.mode(torch.zeros(1).to(torch.int64).to(device)).values)
            else:
                with torch.no_grad():
                    inputs = torch.Tensor(torch.stack(windows)).to(torch.int64).to(device)
                    outputs = model(inputs) # inference the windows
                    predictions.append(torch.mode(torch.round(outputs).squeeze(1)).values) # append the mode of the inference as the prediction of the bio
        
        confmat = BinaryConfusionMatrix().to(device)
        print(confmat(torch.stack(predictions), torch.Tensor(test_target).to(device)))

    if params.model == 'LSTM_CLASSIFY':
#          {add code to instantiate the model, recall model parameters and perform/learn classification}
        print("Testing LSTM model...")

        test_sq = LSTM_split_sequences(test, params.window, params.window)
        test_sq_loss = LSTM_split_sequences(test, params.window + 1, params.window + 1)

        model = torch.load('./'+ params.loadname + '.pt')
        print("begin testing...")
        
        predictions = LSTM_generate(test_sq)
        test_input, test_targets = LSTM_preprocess(test)

        confmat = BinaryConfusionMatrix().to(device)
        print(confmat(torch.Tensor(predictions).to(device),torch.Tensor(test_targets).to(device)))
        print(classification_report(test_targets,predictions))

    if params.model == 'BLIND':
#          {add code to instantiate the model, recall model parameters and perform/learn classification}
        print("Testing blind set...")

        model = torch.load('./FFNN_model.pt')
        [vocab,words,blind] = read_encode("./data/blind.test.txt",vocab,words,[],-1,'FFNN')
        test_input = FFNN_getInput(blind)

        test_input = [torch.Tensor(i) for i in test_input]
        model.eval()
        predictions = []

        for bio in test_input:
            windows = FFNN_split_window(params.window, bio) # split each testing bio into windows
            if len(bio) < params.window:
                predictions.append(torch.mode(torch.zeros(1).to(torch.int64).to(device)).values)
            else:
                with torch.no_grad():
                    inputs = torch.Tensor(torch.stack(windows)).to(torch.int64).to(device)
                    outputs = model(inputs) # inference the windows
                    predictions.append(torch.mode(torch.round(outputs).squeeze(1)).values) # append the mode of the inference as the prediction of the bio
        
        predictions = [i.to(torch.int64).tolist() for i in predictions]

        with open('predictions.csv', mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(['id', 'prediction'])
            for i, pred in enumerate(predictions):
                writer.writerow([i, pred])

if __name__ == "__main__":
    main()