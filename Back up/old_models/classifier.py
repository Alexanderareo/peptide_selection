import sys

import preprocess

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

import pandas as pd
import numpy as np
import math

from matplotlib import pyplot as plt

embedding_dim = 100
hidden_num = 32
num_layer = 2
batch_size = 16
learning_rate = 0.00005
dropout = 0.7
bidirectional = True

Letter_dict = {}

data = pd.read_csv("/home/xuyanchao/peptide_selection/datasets/all_data_with_negative_labels.csv",encoding="utf8")
all_data = preprocess.df2list(data,"sequence","type",3,is_origin="Nop")
train_data,test_data = preprocess.data_split(all_data,9000)
Letter_dict = preprocess.build_dict(all_data)
weight_dict = torch.randn(len(Letter_dict)+1,embedding_dim)

def pre_process(data):
    data_process = []
    for i in range(len(data)):
        tmp = []
        for j in range(len(data[i])):
            tmp.append(Letter_dict[data[i][j]])
        data_process.append(tmp)

    return data_process

class PeptideDataset(Dataset):
    def __init__(self,all_data):
        self.target = all_data[1]
        self.sequence = pre_process(all_data[0])

    def __getitem__(self,index):
        sequence = self.sequence[index]
        target = self.target[index]
        length = len(sequence)
        if len(sequence) < 50:
            npi = np.zeros((50 - len(sequence)), dtype=np.int)
            sequence.extend(npi)
        return sequence, target, length

    def __len__(self):
        return len(self.sequence)

def user_func(batch): # 两种padding的顺序
    output = []
    output_label = []
    length = []
    batch = sorted(batch, key=lambda x:x[2],reverse=True)
    for i in batch:
        output.append(i[0])
        output_label.append(i[1])
        length.append(i[2])
    output = torch.tensor(output)
    output_label = torch.tensor(output_label)
    length = torch.tensor(length)

    return output, output_label,length

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.lstm = torch.nn.LSTM(embedding_dim,hidden_num,num_layer,bidirectional=bidirectional,batch_first=True,dropout=dropout)
        self.linear = nn.Sequential(
            nn.Linear(hidden_num*num_layer*(2 if bidirectional == True else 1),16),
            nn.ReLU(inplace=True),
            nn.Linear(16,2),
        )
        self.embedding = torch.nn.Embedding(num_embeddings=len(Letter_dict)+1, embedding_dim=embedding_dim, padding_idx=0,_weight=weight_dict)
        
    def forward(self,x,length):
        # import ipdb; ipdb.set_trace()
        x = self.embedding(x)
        x = pack_padded_sequence(input = x,lengths=length,batch_first=True)
        output, (h_s,h_c) = self.lstm(x)
        h_n = h_s
        h_n = torch.reshape(h_n.transpose(0,1),(-1,hidden_num*num_layer*(2 if bidirectional == True else 1)))
        out = self.linear(h_n)
        return out

def main():
    train_data_set = PeptideDataset(train_data)
    test_data_set = PeptideDataset(test_data)
    train_loader = DataLoader(dataset = train_data_set, batch_size=batch_size, shuffle = False,collate_fn = user_func,drop_last=True)
    test_loader = DataLoader(dataset = test_data_set, batch_size=batch_size, shuffle = False,collate_fn = user_func,drop_last=True)
    device = torch.device("cuda:1")
    model = Net().to(device)
    train(model,200,train_loader,test_loader,device)

def train(model, epoch_num, train_loader,test_loader,device):
    loss_all = 10000
    train_loss = []
    test_loss = []
    for epoch in range(epoch_num):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        count = 0
        train_acc, test_acc = 0, 0
        acc = 0
        model.train()
        for idx, batch in enumerate(train_loader):
            text = torch.LongTensor(batch[0])
            label = batch[1]
            length = batch[2]
            text = text.to(device)
            label = label.to(device)
            length = length.to(device)
            out = model(text,length)
            optimizer.zero_grad()
            loss = loss_fn(out,label.long())
            # TODO
            # loss1 = torch.nn.functional.binary_cross_entropy(out, label_binary)
            # (loss + theta * loss1).backward()  # TODO argparse
            loss.backward(retain_graph=True)
            optimizer.step()
            acc += sum((torch.argmax(out,1)==label).cpu().numpy())
            accuracy = np.mean((torch.argmax(out,1)==label).cpu().numpy())
            if count % 100==0:
                print(" Epoch:%d,acc:%f"%(epoch,accuracy))
            count+=1
        print("\n Epoch:{} loss:{}, train_acc:{}, totol_acc:{}".format(epoch,loss.item(),accuracy,acc/(count*batch_size)),end=" ")
        train_loss.append(loss)
    
        model.eval()
        predict = list([])
        label_o = list([])
        acc = 0
        for idx, batch in enumerate(test_loader):
            text = torch.LongTensor(batch[0])
            label = batch[1]
            length = batch[2]
            text = text.to(device)
            label = label.to(device)
            length = length.to(device)
            out = model(text,length)
            accuracy = np.mean((torch.argmax(out,1)==label).cpu().numpy())
            acc += accuracy*batch_size
            predict.extend(out.cpu().detach().numpy().reshape(1,-1)[0])
            label_o.extend(label.cpu().detach().numpy())
        print("\n Epoch: %d Accuracy: %f"%(epoch,acc/len(label_o)))
    
if __name__ == "__main__":
    main() 
    
