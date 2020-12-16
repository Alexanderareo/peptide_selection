import sys
import argparse
from datetime import datetime
import preprocess_12_15

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
        self.target = all_data[1]
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
    loss_best = 10000
    train_loss = []
    test_loss = []
    test_r2 = []
    for epoch in range(epoch_num):
        loss_fn = nn.MSELoss()
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
        pred = []
        label = []
        for i in range(len(label_o)):
            if mod =="all":
                label.append(label_o[i])
                pred.append(predict[i])
            else:
                if label_o[i]<=np.log10(8196)-0.01:
                    label.append(label_o[i])
                    pred.append(predict[i])
        mse_result = mean_squared_error(label, pred)
        r2_result = r2_score(label, pred)
        test_loss.append(mse_result)
        test_r2.append(r2_result)
        if mse_result <= loss_best:
            loss_best = mse_result
            print("work")
        # draw_pic_result(log_num,pred,label)
        print("MSE_loss = %f"%(mse_result))
        print("R2_score = %f"%(r2_result))
        print("\r Epoch: %d Best MSE Error: %f"%(epoch,loss_best))
    
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
    train_loader = DataLoader(dataset = train_data_set, batch_size=batch_size, shuffle = False,collate_fn = user_func,drop_last=True)
    test_loader = DataLoader(dataset = test_data_set, batch_size=batch_size, shuffle = False,collate_fn = user_func,drop_last=True)
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
    parser.add_argument("-cn","--case_num",type=int)
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
    case_num = args.case_num
    mod = args.mod
    if mod == "all":
        all_file_path = "datasets/standard_files/data_" + args.input_file_path + "_8047.csv"
        train_path = "datasets/split_data/" + args.input_file_path + "_8047_train.csv"
        test_path = "datasets/split_data/" + args.input_file_path + "_8047_test.csv"
        val_path = "datasets/split_data/" + args.input_file_path + "_8047_val.csv"
    else:
        all_file_path = "datasets/standard_files/positive_sample_" + args.input_file_path +".csv"
        train_path = "datasets/split_data/" + args.input_file_path + "_train.csv"
        test_path = "datasets/split_data/" + args.input_file_path + "_test.csv"
        val_path = "datasets/split_data/" + args.input_file_path + "_val.csv"
    Letter_dict = {}
    data = pd.read_csv(all_file_path,encoding="utf8")
    data_train = pd.read_csv(train_path,encoding="utf8")
    data_test = pd.read_csv(test_path,encoding="utf8")
    data_val = pd.read_csv(val_path,encoding="utf8")
    
    data.reset_index(drop=True)
    all_data = preprocess_12_15.df2list(data,"sequence","MIC","type",ngram_num,log_num)
    Letter_dict = preprocess_12_15.build_dict(all_data)
    # train_data = preprocess.df2list(data_train,"sequence","MIC","type",ngram_num,log_num)
    # test_data = preprocess.df2list(data_test,"sequence","MIC","type",ngram_num,log_num)
    # val_data = preprocess.df2list(data_val,"sequence","MIC","type",ngram_num,log_num)
    weight_dict = torch.randn(len(Letter_dict)+1,embedding_dim)
    train_data,test_data = preprocess_12_15.data_split(all_data,case_num*split_rate)
    main()