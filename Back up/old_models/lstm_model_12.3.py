import sys
import argparse
from preprocess import filter_data_with_str,duplicated_process,log_process,df2list,data_split,build_dict

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
from sklearn.metrics import mean_squared_error

import tqdm

from matplotlib import pyplot as plt

embedding_dim = 50
hidden_num = 32
num_layer = 2
batch_size = 16
learning_rate = 0.0001
dropout = 0.7
bidirectional = True

Letter_dict = {}

data = pd.read_csv("/home/xuyanchao/peptide_selection/datasets/old_files/test.csv",encoding="utf8")
all_data = df2list(data,"sequence","MIC",1,2)
train_data,test_data = data_split(all_data,9000)
Letter_dict = build_dict(all_data)
weight_dict = torch.randn(len(Letter_dict)+1,embedding_dim)

def preprocess(data):
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
        self.target = all_data[1]
        self.sequence = preprocess(all_data[0])

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
            nn.Linear(16,1),
        )
        self.embedding = torch.nn.Embedding(num_embeddings=len(Letter_dict)+1,embedding_dim=embedding_dim, padding_idx=0,_weight=weight_dict)
        # self.layer_norm = LayerNorm(hidden_num*num_layer*(2 if bidirectional == True else 1))
    def forward(self,x,length):
        # import ipdb; ipdb.set_trace()
        x = self.embedding(x)
        x = pack_padded_sequence(input = x,lengths=length,batch_first=True)
        output, (h_s,h_c) = self.lstm(x)
        h_n = h_s
        h_n = torch.reshape(h_n.transpose(0,1),(-1,hidden_num*num_layer*(2 if bidirectional == True else 1)))
        # h_n = self.layer_norm(h_n)
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
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
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
            loss = loss_fn(out,torch.reshape(label,(batch_size,1)))
            loss_all += loss
            count+=1
            # TODO
            # loss1 = torch.nn.functional.binary_cross_entropy(out, label_binary)
            # (loss + theta * loss1).backward()  # TODO argparse
            if count % 100 == 0:
                print("\r Epoch:%d,Loss:%f"%(epoch,loss_all/count))
            loss.backward(retain_graph=True)
            optimizer.step()
        train_loss.append(loss)
    
        model.eval()
        predict = list([])
        label_o = list([])
        for idx, batch in enumerate(test_loader):
            text = torch.LongTensor(batch[0])
            label = batch[1]
            length = batch[2]
            text = text.to(device)
            label = label.to(device)
            length = length.to(device)
            out = model(text,length)
            # predict_list = out.cpu().detach().numpy()
            # label_list = label.cpu().detach().numpy()
            # for i in range(len(predict_list)):
            #     if int(label_list[i])!=11:
            #         predict.append(float(predict_list[i][0]))
            #         label_o.append(float(label_list[i]))
            predict.extend(out.cpu().detach().numpy().reshape(1,-1)[0])
            label_o.extend(label.cpu().detach().numpy())

        mse_result = mean_squared_error(predict, label_o)
        test_loss.append(mse_result)
        if mse_result <= loss_all:
            loss_all = mse_result
            print("work")
        if mse_result < 11:
            x = np.arange(-1,5,1)
            plt.xlim((-1,4.5))
            plt.ylim((-1,4.5))
            plt.xlabel("Predict")
            plt.ylabel("Origin")
            plt.plot(x,x+0,'--',label="y=x")
            plt.scatter(predict,label_o,marker='o',color = 'blue', s=5)
            plt.legend(loc='upper left')
            plt.savefig("result.png")
            plt.clf()
        print("MSE_loss = %f"%(mse_result))
        print("\r Epoch: %d Best MSE Error: %f"%(epoch,loss_all))
    
    plt.xlabel("Epoch")
    plt.ylabel("MSE_loss")
    plt.plot(train_loss,label="train_loss",marker=".",color="blue",)
    plt.plot(test_loss,label="test_loss",marker="+",color="goldenrod")
    plt.savefig("loss.png")
    plt.clf()

if __name__ == "__main__":
    main() 