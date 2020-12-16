import sys
import argparse
from datetime import datetime
import preprocess

import copy
import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch.utils.data as Data
from torch.nn import LayerNorm

import copy
import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch.utils.data as Data
from torch.nn import LayerNorm

import pandas as pd
import numpy as np
import math

from matplotlib import pyplot as plt

def build_index(data):
    """
    Preprocess
    负责词表的映射
    建立词向量
    input: data 序列
    output: 映射完的数据
    """
    data_process = []
    for i in range(len(data)):
        tmp = []
        for j in range(len(data[i])):
            tmp.append(Letter_dict[data[i][j]])
        data_process.append(tmp)
        # if len(tmp)<=2:
        #     print(data[i])
    return data_process


class PeptideDataset(Dataset):

    def __init__(self, all_data): 
        self.target = all_data[2]
        self.sequence = build_index(all_data[0])

    def __getitem__(self, index):
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
        self.embedding = torch.nn.Embedding(num_embeddings=len(Letter_dict)+1,embedding_dim=embedding_dim, padding_idx=0,_weight=weight_dict)
    def forward(self,x,length):
        x = self.embedding(x)
        x = pack_padded_sequence(input = x,lengths=length,batch_first=True)
        output, (h_s,h_c) = self.lstm(x)
        h_n = h_s
        h_n = torch.reshape(h_n.transpose(0,1),(-1,hidden_num*num_layer*(2 if bidirectional == True else 1)))
        out = self.linear(h_n)
        return out

def train(model, epoch_num, train_loader,test_loader,device):
    loss_best = 0
    train_loss = []
    test_loss = []
    for epoch in range(epoch_num):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        loss_all = 0
        count = 0
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
            loss.backward(retain_graph=True)
            optimizer.step()
            loss_all += loss
            count+=1

            if count % 100 == 0:
                print("\r Epoch:%d,Loss:%f"%(epoch,loss_all/count))
        train_loss.append(loss)
    
        model.eval()
        predict = list([])
        label_o = list([])
        TP = 1
        FP = 1
        FN = 1
        TN = 1
        for idx, batch in enumerate(test_loader):
            text = torch.LongTensor(batch[0])
            label = batch[1]
            length = batch[2]
            text = text.to(device)
            label = label.to(device)
            length = length.to(device)
            out = model(text,length)
            label_out = torch.argmax(out,1)
            predict.extend(label_out.cpu().detach().numpy().reshape(1,-1)[0])
            label_o.extend(label.cpu().detach().numpy())
        for i in range(len(predict)):
            if predict[i]==1 and label_o[i]==1:
                TP += 1
            elif predict[i]==1 and label_o[i]==0:
                FN += 1
            elif predict[i]==0 and label_o[i]==0:
                TN += 1
            else:
                FP += 1
        accuracy = (TP+TN)/(TP+TN+FN+FP)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        specificity = TN/(TN+FP)
        F1 = 2*precision*recall/(precision+recall)

        if F1 >= loss_best:
            loss_best = F1
            print("work")
        test_loss.append(F1)
        print("Accuracy = %f Precision = %f Recall = %f Specificity = %f F1 = %f"%(accuracy,precision,recall,specificity,F1))
        print("\r Epoch: %d Best F1: %f"%(epoch,loss_best))


def main():
    
    train_data_set = PeptideDataset(train_data)
    test_data_set = PeptideDataset(test_data)
    train_loader = DataLoader(dataset = train_data_set, batch_size=batch_size, shuffle = False,collate_fn = user_func,drop_last=True)
    test_loader = DataLoader(dataset = test_data_set, batch_size=batch_size, shuffle = False,collate_fn = user_func,drop_last=True)
    device = torch.device("cpu")
    model = Net().to(device)
    train(model,epoch_num,train_loader,test_loader,device)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-ed","--embedding_dim",type=int, default=50)
    parser.add_argument("-hn","--hidden_num",type=int, default=32)
    parser.add_argument("-nl","--num_layer",type=int,default=2)
    parser.add_argument("-bs","--batch_size",type=int,default=16)
    parser.add_argument("-lr","--learning_rate",type=float,default=1e-4)
    parser.add_argument("-dp","--dropout",type=float,default=0.7)
    parser.add_argument("-bd","--bidirectional",type=bool,default=True)
    parser.add_argument("-f","--input_file_path",type=str)
    parser.add_argument("-n","--ngram_num",type=int,default=1)
    parser.add_argument("-e","--epoch_num",type=int,default=200)
    parser.add_argument("-sr","--split_rate",type=float,default=0.8)
    args = parser.parse_args()
    print(vars(args))
    embedding_dim = args.embedding_dim
    hidden_num = args.hidden_num
    num_layer = args.num_layer
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    dropout = args.dropout
    bidirectional = args.bidirectional
    input_file_path = "datasets/standard_files/" + args.input_file_path
    ngram_num = args.ngram_num
    epoch_num = args.epoch_num
    split_rate = args.split_rate

    Letter_dict = {}
    data = pd.read_csv(input_file_path,encoding="utf8")
    data = data.sample(frac=1)
    data.reset_index(drop=True)
    all_data = preprocess.df2list(data,"sequence","MIC","type",ngram_num,log_num=10)
    Letter_dict = preprocess.build_dict(all_data)
    weight_dict = torch.randn(len(Letter_dict)+1,embedding_dim)
    num1 = int(input_file_path[-13:-9])
    # num1 = 0
    num2 = 8407
    train_data,test_data = preprocess.data_split(all_data,int((num1+num2)*split_rate))
    main()