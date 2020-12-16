import pandas as pd
import numpy as np
import string
import re
import sys
import itertools
data = pd.read_csv("/home/xyc/peptide_selection/datasets/grampa.csv",encoding="utf8")

AALetter=["A","R","N","D","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
PaddingLetter = ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]","<S>","<T>"]

def generate_bert_dict(Letter_dict, ngram_num, dict_name, pad_dict):
    generate_dict = ["".join(i) for i in itertools.product(*[Letter_dict for _ in range(ngram_num)])]
    file = open(dict_name+".txt","w",encoding="utf8")
    for i in pad_dict:
        file.write(i+"\n")
    for i in generate_dict:
        file.write(i+"\n")
    file.close()
    return generate_dict

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

def duplicated_process(col_name,data):
    """
    dataframe去重
    input: 
        col_name: 去重的列
        data: dataframe
    output:
        filter_data: 去重完的数据
    """
    filter_data = data.drop_duplicates([col_name])
    return filter_data

def log_process(col_name, source1, source2, data):
    """
    对数底的转换(这里是将原先的ug单位转换为uM单位)
    input:
        col_name: 需要进行数值转换的列
        source1: 需要的ug的数据
        source2: 需要的分子量的数据
    output:
        data: 转换完的数据
    """
    for i in range(data.shape[0]):
        data.loc[i,col_name] = np.log10((data.loc[i,source1]*1000)/data.loc[i, source2])
    return data

def df2list(data_filter,varible,label,l_type,ngram_num,log_num=2):
    """
    将dataframe转换为需要的list对象 格式为[["sequence"],["label"]],
    并将其中的完整list转换为n_gram的形式
    input:
        data_filter: 原始dataframe
        variable: sequence序列
        label: 抗菌性的结果
        n_gram: n_gram number
    output:
        all_data: 格式化并且分完词的数据
    """
    all_data = [[],[],[]]
    for i in data_filter.iterrows():
        if len(list(i[1][varible]))<=50:
            all_data[0].append(create_ngram_list(i[1][varible], ngram_num))
        else:
            all_data[0].append(create_ngram_list(i[1][varible][0:49], ngram_num))
        if log_num == 2:
            all_data[1].append(float(np.log2(float(i[1][label]))))
            all_data[2].append(i[1][l_type])
        elif log_num == 10:
            all_data[1].append(float(np.log10(float(i[1][label]))))
            all_data[2].append(i[1][l_type])
        else:
            all_data[1].append(i[1][label])
            all_data[2].append(i[1][l_type])
    return all_data

def data_split(dataset_input,index):
    """
    切分训练集和测试集
    input：
        dataset_input: 原始数据集
        index: 切分训练集和测试集的位置标签
    output:
        data_train和data_test: 测试集和训练集
    """
    data_test = [[],[],[]]
    data_train = [[],[],[]]
    for i in range(len(dataset_input[0])):
        if i >= index:                                                                  
            data_test[0].append(dataset_input[0][i])
            data_test[1].append(dataset_input[1][i])
            data_test[2].append(dataset_input[2][i])
        else:                                                              
            data_train[0].append(dataset_input[0][i])
            data_train[1].append(dataset_input[1][i])
            data_train[2].append(dataset_input[2][i])
    return data_train,data_test

def create_ngram_list(input_list, ngram_num):
    """
    建立n分词的列表
    input:
        input_list: 需要切分的list
        ngram_num:
    output:
        ngram_list: 切好的list
    """
    ngram_list = []
    if len(input_list)<ngram_num:
        ngram_list = [x for x in input_list]
    else:
        for i in range(len(input_list)-ngram_num+1):
            ngram_list.append(input_list[i:i+ngram_num])
    return ngram_list

def build_dict(data):
    """
    根据data建立词表
    input:
        data
    output:
        Letter_dict
    """
    gram_list = []
    Letter_dict = {}
    for i in range(len(data[0])):
        for j in data[0][i]:
          gram_list.append(j)  
    gram_list = list(set(gram_list))
    for i in range(len(gram_list)):
        Letter_dict[gram_list[i]] = i+1
    return Letter_dict

if __name__ == '__main__':
    # data_filter = filter_data_with_str("bacterium","aureus",data)
    # data_filter = duplicated_process("sequence",data_filter)
    # print(build_dict([[['abc', 'bcd', 'cdo', 'dod', 'odf', 'dfg', 'fgv']],[10]]))
    print(generate_bert_dict(AALetter, 1, "ngram_1",PaddingLetter)[:20])
