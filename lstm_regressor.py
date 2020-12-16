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

import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error, r2_score

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
        tmp_t = []
        tmp_s = []
        tmp_p = []
        for i in range(len(all_data)):
            tmp_s.append(all_data[i][0])
            tmp_t.append(all_data[i][1])
            tmp_p.append(all_data[i][2])
        self.target = tmp_t
        self.sequence = build_index(tmp_s)
        self.data_type = tmp_p

    def __getitem__(self, index):
        sequence = self.sequence[index]
        target = self.target[index]
        data_type = self.data_type[index]
        length = len(sequence)
        if len(sequence) < 50:
            npi = np.zeros((50 - len(sequence)), dtype=np.int)
            sequence.extend(npi)
        return sequence, target, data_type, length

    def __len__(self):
        return len(self.sequence)
        
def user_func(batch): # 两种padding的顺序
    output = []
    output_label = []
    output_type = []
    length = []
    batch = sorted(batch, key=lambda x:x[3],reverse=True)
    for i in batch:
        output.append(i[0])
        output_label.append(i[1])
        output_type.append(i[2])
        length.append(i[3])
    output = torch.tensor(output)
    output_label = torch.tensor(output_label)
    output_type = torch.tensor(output_type)
    length = torch.tensor(length)

    return output, output_label, output_type, length

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
        x = self.embedding(x)
        x = pack_padded_sequence(input = x,lengths=length,batch_first=True)
        output, (h_s,h_c) = self.lstm(x)
        h_n = h_s
        h_n = torch.reshape(h_n.transpose(0,1),(-1,hidden_num*num_layer*(2 if bidirectional == True else 1)))
        # h_n = self.layer_norm(h_n)
        out = self.linear(h_n)
        return out

def train(model, epoch_num, train_loader,test_loader,device):
    loss_best_all = 10000
    loss_best_pos = 10000
    r2_best_all = -5
    r2_best_pos = -5
    train_loss = []
    test_loss_all = []
    test_loss_pos = []
    test_r2_all = []
    test_r2_pos = []
    for epoch in range(epoch_num):
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        loss_all = 0
        count = 0
        model.train()
        for idx, batch in enumerate(train_loader):
            text = torch.LongTensor(batch[0])
            label = batch[1]
            length = batch[3]
            text = text.to(device)
            label = label.to(device)
            # length = length.to(device)
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
            length = batch[3]
            text = text.to(device)
            label = label.to(device)
            # length = length.to(device)
            out = model(text,length)

            predict.extend(out.cpu().detach().numpy().reshape(1,-1)[0])
            label_o.extend(label.cpu().detach().numpy())
        pred = []
        label = []
        for i in range(len(label_o)):
            if label_o[i]<=np.log10(8196)-0.01:
                label.append(label_o[i])
                pred.append(predict[i])
        mse_result_all = mean_squared_error(label_o,predict)
        mse_result_pos = mean_squared_error(label, pred)
        r2_result_all = r2_score(label_o, predict)
        r2_result_pos = r2_score(label, pred)
        test_loss_all.append(mse_result_all)
        test_loss_pos.append(mse_result_pos)
        test_r2_all.append(r2_result_all)
        test_r2_pos.append(r2_result_pos)
        if mse_result_all <= loss_best_all:
            loss_best_all = mse_result_all
        if mse_result_pos <= loss_best_pos:
            loss_best_pos = mse_result_pos
        if r2_result_all >= r2_best_all:
            r2_best_all = r2_result_all
        if r2_result_pos >= r2_best_pos:
            r2_best_pos = r2_result_pos
        # draw_pic_result(log_num,pred,label)
        print("MSE_loss_all = %f"%(mse_result_all))
        print("R2_score_all = %f"%(r2_result_all))
        print("MSE_loss_pos = %f"%(mse_result_pos))
        print("R2_score_pos = %f"%(r2_result_pos))
        print("\r Epoch: %d Best MSE Error all %f ; Best MSE Error pos: %f ; Best R2 Error all: %f ; Best R2 Error pos: %f ;"\
            %(epoch,loss_best_all,loss_best_pos,r2_best_all,r2_best_pos))
    
    # draw_pic_loss(train_loss,test_loss,test_r2,desc="mse_loss")


def draw_pic_result(log_num,predict,label_o):
    if log_num == 2:
        x_min = y_min = -2
        x_max = y_max = 14
        bias = 2
    elif log_num == 10:
        x_min = y_min = -1
        x_max = y_max = 4.5
        bias = 1
    else:
        x_min = y_min = 0
        x_max = y_max = 8196
        bias = 64
    x = np.arange(x_min-bias,x_max+bias,1)
    plt.xlim((x_min,x_max))
    plt.ylim((y_min,y_max))
    plt.xlabel("Predict")
    plt.ylabel("Origin")
    plt.plot(x,x+0,'--',label="y=x")
    plt.plot(x,x+bias,'--',label="y=x+%d"%(bias))
    plt.plot(x,x-bias,'--',label="y=x-%d"%(bias))
    plt.scatter(predict,label_o,marker='o',color = 'blue', s=7)
    plt.legend(loc='upper left')
    date_time = datetime.now().strftime('%Y-%m-%d')
    plt.savefig("result_pic/regress/%s_result_6760_P.png"%(date_time))
    plt.clf()

def draw_pic_loss(train_loss,test_loss,test_r2,desc="mse_r2_loss"):

    plt.xlabel("Epoch")
    plt.ylabel("MSE_loss")
    plt.plot(train_loss,label="train_loss",marker=".",color="blue",)
    plt.plot(test_loss,label="test_loss",marker="+",color="goldenrod")
    plt.plot(test_r2,label="r2_loss",marker=".",color="red",)
    date_time = datetime.now().strftime('%Y-%m-%d')
    plt.savefig("result_pic/regress/%s_%s_6760_P.png"%(date_time,desc))
    plt.clf()

def main():
    
    train_data_set = PeptideDataset(train_data)
    test_data_set = PeptideDataset(test_data)
    val_data_set = PeptideDataset(val_data)
    train_loader = DataLoader(dataset = train_data_set, batch_size=batch_size, shuffle = False,collate_fn = user_func,drop_last=True)
    test_loader = DataLoader(dataset = test_data_set, batch_size=batch_size, shuffle = False,collate_fn = user_func,drop_last=True)
    val_loader = DataLoader(dataset = val_data_set, batch_size=batch_size, shuffle = False,collate_fn = user_func,drop_last=True)
    device = torch.device("cuda:1")
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
    parser.add_argument("-lg","--log_num",type=int,default=10)
    parser.add_argument("-sr","--split_rate",type=float,default=0.8)
    # parser.add_argument("-cn","--case_num",type=int,default=10000)
    parser.add_argument("-md","--mod",type=str,default="all") # all 加入负样例 / 其他(pos) 不加入负样例
    args = parser.parse_args()
    print(vars(args))
    embedding_dim = args.embedding_dim
    hidden_num = args.hidden_num
    num_layer = args.num_layer
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    dropout = args.dropout
    bidirectional = args.bidirectional
    ngram_num = args.ngram_num
    epoch_num = args.epoch_num
    log_num = args.log_num
    split_rate = args.split_rate
    # case_num = args.case_num
    mod = args.mod
    if mod == "all":
        all_file_path = "datasets/standard_files/data_" + args.input_file_path + "_8047.csv"
        train_path = "datasets/split_data/" + args.input_file_path + "_8047_train.csv"
        test_path = "datasets/split_data/" + args.input_file_path + "_8047_test.csv"
        val_path = "datasets/split_data/" + args.input_file_path + "_8047_val.csv"
    else:
        all_file_path = "datasets/standard_files/positive_samples_" + args.input_file_path +".csv"
        train_path = "datasets/split_data/" + args.input_file_path + "_train.csv"
        test_path = "datasets/split_data/" + args.input_file_path + "_test.csv"
        val_path = "datasets/split_data/" + args.input_file_path + "_val.csv"
    Letter_dict = {}
    data = pd.read_csv(all_file_path,encoding="utf8")
    data_train = pd.read_csv(train_path,encoding="utf8")
    data_test = pd.read_csv(test_path,encoding="utf8")
    data_val = pd.read_csv(val_path,encoding="utf8")
    
    data.reset_index(drop=True)
    all_data = preprocess.df2list(data,"sequence","MIC","type",ngram_num,log_num)
    Letter_dict = preprocess.build_dict(all_data)
    train_data = preprocess.df2list(data_train,"sequence","MIC","type",ngram_num,log_num)
    test_data = preprocess.df2list(data_test,"sequence","MIC","type",ngram_num,log_num)
    val_data = preprocess.df2list(data_val,"sequence","MIC","type",ngram_num,log_num)
    weight_dict = torch.randn(len(Letter_dict)+1,embedding_dim)
    # train_data,test_data = preprocess.data_split(all_data,case_num*split_rate)
    main()