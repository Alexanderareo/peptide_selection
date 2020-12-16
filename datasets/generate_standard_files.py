import pandas as pd
import sys
import numpy as np
import math
import argparse

def filter_data_with_str(col_name,str,data):
    """
    Input
    col_name: 筛选列
    str: 需要筛选包含的字符串
    data: 需要处理的数据
    Output
    filter_data: 处理完的数据
    """
    # 效率更高
    # filter_data = data[data[col_name].str.contains(str)]
    bool_filter = data[col_name].str.contains(str)
    filter_data = data[bool_filter]
    return filter_data

def generate_all_peptides(data):
    """
    去重+取均值
    """
    data_all = [[],[],[]]
    for i in data["sequence"].unique():
        data_all[0].append(i)
        log_num = 0
        count = 0
        for i in data[data["sequence"]==i]["value"]:
            log_num += math.pow(10,i)
            count+=1
        data_all[1].append(float(log_num/count))
        data_all[2].append(1)

    data_all = list(map(list, zip(*data_all))) # 转置
    data = pd.DataFrame(data=data_all, columns=["sequence","MIC","type"])
    return data

def data2csv(data,file_name):
    data.to_csv(file_name,encoding="utf8",index=False)

def generate_negative_data(negative_file_path="filtered_negative.csv",output_path="negative_samples_8047.csv"):    

    data_negative = pd.read_csv(negative_file_path,encoding="utf8")
    data_negative = data_negative[~data_negative["Sequence"].str.contains("B|X|Z|O|U")]
    data_negative.reset_index(drop=True, inplace=True)
    data = pd.DataFrame(columns=["sequence","MIC","type"])
    for i in range(data_negative.shape[0]):
        data = data.append({"sequence":data_negative["Sequence"][i],"MIC":8196,"type":0},ignore_index=True)
    # print(data.describe())
    # sys.exit()
    data2csv(data,"standard_files/"+output_path)


def concat_datasets(positive_file,negative_file):

    data_concat = pd.concat([positive_file,negative_file],ignore_index=True, axis=0) # 默认纵向合并0 横向合并1
    data_concat = data_concat.sample(frac=1)
    data_concat.reset_index(drop=True, inplace=True)
    return data_concat

if __name__ == '__main__':
    # data = pd.read_csv("grampa.csv",encoding="utf8")
    # data = filter_data_with_str("bacterium","aureus",data)
    # data = generate_all_peptides(data)
    # data2csv(data,"standard_files/positive_samples_6760.csv")
    # print(data.describe())
    # generate_negative_data()
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--pos_file",type=str, default="null",help="positive_file_path")
    parser.add_argument("-n","--neg_file",type=str, default="null",help="negative_file_path")
    parser.add_argument("-o","--output_path",type=str,default="null",help="output_file_path")
    args = parser.parse_args()
    print(args.pos_file)
    pos_path = "standard_files/"+args.pos_file
    neg_path = "standard_files/"+args.neg_file
    output = "standard_files/"+args.output_path
    pos_file = pd.read_csv(pos_path,encoding="utf8")
    neg_file = pd.read_csv(neg_path,encoding="utf8")
    data = concat_datasets(pos_file,neg_file)
    data2csv(data,output)