import sys

from preprocess import filter_data_with_str,duplicated_process,log_process,df2list,data_split,build_dict

import copy
import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import torch.utils.data as Data

import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error

import tqdm

from matplotlib import pyplot as plt

AALetter=["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
# Letter_dict = {"A":2,"R":3,"N":4,"D":5,"C":6,"E":7,"Q":8,"G":9,"H":10,"I":11,"L":12,"K":13,"M":14,"F":15,"P":16,"S":17,"T":18,"W":19,"Y":20,"V":21}

embedding_dim = 50
hidden_num = 32
num_layer = 2
batch_size = 16
learning_rate = 0.0001
dropout = 0.8
bidirectional = True
Letter_dict = {}

# data = pd.read_csv("/home/xuyanchao/peptide_selection/datasets/grampa.csv",encoding="utf8")
# data_test = pd.read_csv("/home/xuyanchao/server_test/origin_data/Bacteria_latest_11.16.csv",encoding="utf8")
# data_test = log_process("result","staphylococcus aureus(ug/mL)","molecular weight",data_test)
# data_filter = filter_data_with_str("bacterium","aureus",data)
# data_filter = duplicated_process("sequence",data_filter)

# all_data = df2list(data_filter,"sequence","value",3)
# print(len(all_data[0]))
# data_test_all = df2list(data_test,"sequence","result",3)
# train_data,test_data = data_split(all_data,4200)
# Letter_dict = build_dict(all_data)
# weight_dict = torch.randn(len(Letter_dict)+1,embedding_dim) # 建立随机的权重矩阵 [词表长度, 词向量维度]

data = pd.read_csv("/home/xuyanchao/peptide_selection/datasets/all_data_with_negative.csv",encoding="utf8")
all_data = df2list(data,"sequence","MIC",3,True)
train_data,test_data = data_split(all_data,9000)
Letter_dict = build_dict(all_data)
weight_dict = torch.randn(len(Letter_dict)+1,embedding_dim) # 建立随机的权重矩阵 [词表长度, 词向量维度]

# all_data_exclude_test = [[],[]]
# for i in range(len(all_data[0])):
#     if all_data[0][i] not in data_test_all[0]:
#         all_data_exclude_test[0].append(all_data[0][i])
#         all_data_exclude_test[1].append(all_data[1][i])

# all_data = all_data_exclude_test
# print(Letter_dict)
# print(train_data[0][0])
# sys.exit()


"""
# TODO
0. git管理
1. 放弃LSTM，跑一个bag-of-ngram模型, ngram max length=(3,4,5), 看整体的MSE和散点图
2. 在LSTM的框架下, 对于模型来说：
 2.1 重写dataloader，每一次一个batch一半正样本，一半负样本
 2.2 不要用全连接层，求mean或者max, 得出结果 (用下unpack）
 2.3 加dropout进去
 2.4 ADAM换成AdamW
 2.5 add clip_gram_norm
 2.6 加上Layer normalization
3. 回归问题本身, 负样本
 3.1 用两个loss，一个还是mse 但只负责正样本部分，另一个是binary cross-entropy负责正负样本的分类
 3.2 一个问题？找一下相关的文献，有没有类似的问题 分类回归一次做的。google + kaggle上找一下
"""


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
    # max_length = len(batch[0][0]) + 10
    # max_length = 50
    for i in batch:
        # if len(i[0])<max_length:
        #     npi = np.zeros((max_length-len(i[0])),dtype=np.int)
        #     i[0].extend(npi)
        output.append(i[0])
        output_label.append(i[1])
        length.append(i[2])
    output = torch.tensor(output)
    output_label = torch.tensor(output_label)
    length = torch.tensor(length)
    # print(output)
    # print(length)
    # sys.exit()
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
        
    def forward(self,x,length):
        import ipdb; ipdb.set_trace()
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
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    loss_all = 10000
    
    for epoch in range(200):
        MSE = 0
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
            # TODO
            loss1 = torch.nn.functional.binary_cross_entropy(out, label_binary)
            (loss + theta * loss1).backward()  # TODO argparse
            if count%100==0:
                print("\r Epoch:%d,Loss:%f"%(epoch,loss))
            loss.backward(retain_graph=True)
            optimizer.step()
            count+=1
    
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
            predict.extend(out.cpu().detach().numpy())
            label_o.extend(label.cpu().detach().numpy())

        mse_result = mean_squared_error(predict, label_o)
        if mse_result<=loss_all:
            loss_all = mse_result
        if mse_result < 1.95:
            x = np.arange(-3,14,1)
            plt.xlim((-3.5,14.5))
            plt.ylim((-3.5,14.5))
            plt.xlabel("Predict")
            plt.ylabel("Origin")
            plt.plot(x,x+0,'--',label="y=x")
            plt.scatter(predict,label_o,marker='o',color = 'blue', s=30)
            plt.legend(loc='upper left')
            plt.savefig("result.png")
            plt.clf()

        print("\r Epoch: %d Best MSE Error: %f"%(epoch,loss_all))

if __name__ == "__main__":
    main()

