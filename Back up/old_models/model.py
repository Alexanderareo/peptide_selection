from preprocess import filter_data_with_str,duplicated_process

import copy
import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import DataLoader,Dataset
import torch.utils.data as Data
import pandas as pd
import numpy as np

from torchtext import datasets
from torchtext import data
from torchtext.data import Field
from torchtext.data import Dataset
from torchtext.data import Example
from torchtext.data import 

import tqdm

# print(list("python"))
AALetter=["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
Letter_dict = {"A":2,"R":3,"N":4,"D":5,"C":6,"E":7,"Q":8,"G":9,"H":10,"I":11,"L":12,"K":13,"M":14,"F":15,"P":16,"S":17,"T":18,"W":19,"Y":20,"V":21}
# 目前来看都在这20个字母内

data = pd.read_csv("/home/xuyanchao/peptide_selection/datasets/grampa.csv",encoding="utf8")
data_filter = filter_data_with_str("bacterium","aureus",data)
data_filter = duplicated_process("sequence",data_filter)

all_data = [[],[]]
for i in data_filter.iterrows():
    if len(list(i[1]["sequence"]))<=50:
        all_data[0].append(list(i[1]["sequence"]))
    else:
        all_data[0].append(list(i[1]["sequence"])[0:49])
    all_data[1].append(float(i[1]["value"]))

print(count)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.lstm = torch.nn.LSTM(50,64,2,bidirectional=True)
        self.linear = nn.Sequential(
            nn.Linear(256,32),
            nn.ReLU(inplace=True),
            nn.Linear(32,1),
        )
        self.embedding = torch.nn.Embedding(num_embeddings=22,embedding_dim=50, padding_idx=1)
        
    def forward(self,x):
        x = self.embedding(x)

        output, hidden = self.lstm(x)

        h_n = hidden[1]
        h_n = torch.reshape(h_n.transpose(0,1),(-1,256))
        out = self.linear(h_n)
        return out


dataset = {}
path = "dataset/"
TEXT = Field(sequential=True,use_vocab=True)
LABEL = Field(sequential=False,use_vocab=False)
FIELDS = [("sequence", TEXT),("MIC", LABEL)]

all_data = list(zip(all_data[0], all_data[1])) # 为啥要加这一句
# zip可以把一行的多个对象打包为1个tuple
examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS),all_data))
dataset = Dataset(examples, fields = FIELDS)

TEXT.build_vocab(dataset)
# for i,example in enumerate(dataset):
#     print(example.MIC)
train, val = dataset.split(split_ratio=0.85)
train_iter, val_iter = BucketIterator.splits(
    (train,val),
    batch_sizes=(16,16),
    device=-1,
    sort_within_batch=False,
    repeat=False,
    shuffle=False
)

train_y = []
val_y = []
for i,data in enumerate(train):
    train_y.append([data.MIC])
for i,data in enumerate(val):
    val_y.append([data.MIC])


# device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
model = Net().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.00005)
best_model_weights = copy.deepcopy(model.state_dict())

best_acc_test = 0

# for epoch in range(50):
#     MSE = 0
#     count = 0
#     train_acc, train_loss = 0, 0
#     val_acc, val_loss = 0, 0
#     for idx, batch in enumerate(train_iter):
#         text = batch.sequence
#         label = torch.tensor(train_y[count:count+len(batch)])
#         text = text.to(device)
#         label = label.to(device)
#         optimizer.zero_grad()
#         out = model(text)
#         loss = loss_fn(out,label.float())
#         print("\rEpoch:%d,Loss:%f"%(epoch,loss))
#         loss.backward(retain_graph=True)
#         optimizer.step()
#         count += len(batch)
        
            
    # print("\n opech:{} loss:{}, train_acc:{}, totol_acc:{}".format(epoch,loss.item(),accuracy,acc/count),end=" ")

    # acc = 0
    # with torch.no_grad():
    #     for idx, batch in enumerate(val_iter):
    #         text, label = batch.sequence, batch.MIC
    #         text = text.to(device)
    #         label = label.to(device)
    #         out = model(text)
    #         loss = loss_fn(out, label.float())

    # if acc/batch_num > best_acc_test:
    #     best_model_weights = copy.deepcopy(model.state_dict())
    #     best_acc_test = acc/batch_num
    # print("\n opech:{} loss:{}, test_acc:{}, totol_acc:{}".format(epoch,loss.item(),accuracy,acc/batch_num),end=" ")

            
# 保存模型
# torch.save(best_model_weights, 'results/best.pth')



    
    


