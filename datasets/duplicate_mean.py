import pandas as pd
import sys
import numpy as np
import math

def gernerate_negative_data(positive_file_path="grampa_filtered_with_origin_result.csv",negative_file_path="filtered_negative.csv",output_path="all_data_with_negative_8096.csv"):
    data = pd.read_csv(positive_file_path, encoding="utf8")
    # data_all = [[],[]]
    # for i in data["sequence"].unique():
    #     data_all[0].append(i)
    #     data_all[1].append(data[data["sequence"]==i]["origin"].mean())

    # data_all = list(map(list, zip(*data_all))) # 转置
    # data = pd.DataFrame(data=data_all, columns=["sequence","MIC"])
    data_negative = pd.read_csv(negative_file_path,encoding="utf8")
    data_negative = data_negative[~data_negative["Sequence"].str.contains("B|X|Z|O|U")]
    data_negative.reset_index(drop=True, inplace=True)
   
    for i in range(data_negative.shape[0]):
        data = data.append({"sequence":data_negative["Sequence"][i],"MIC":8096},ignore_index=True)
        # print({"sequence":data_negative["Sequence"][i],"MIC":2048})

    data = data.sample(frac=1)
    data.reset_index(drop=True, inplace=True)
    data.to_csv(output_path,encoding="utf8")

def generate_all_peptides():
    data = pd.read_csv("grampa.csv",encoding="utf8")
    data_all = [[],[]]
    for i in data["sequence"].unique():
        data_all[0].append(i)
        log_num = 0
        count = 0
        for i in data[data["sequence"]==i]["value"]:
            log_num += math.pow(10,i)
            count+=1
        data_all[1].append(float(log_num/count))
        print(log_num/count)
        sys.exit()

    data_all = list(map(list, zip(*data_all))) # 转置
    data = pd.DataFrame(data=data_all, columns=["sequence","MIC"])
    data.to_csv("grampa_all_data_unique_with_mean.csv",encoding="utf8")
        
def generate_classifier_data(positive_file_path="grampa_filtered_with_origin_result.csv",\
    negative_file_path="filtered_negative.csv",output_path="all_data_with_negative_labels.csv"):
    data_positive = pd.read_csv(positive_file_path, encoding="utf8")
    dataframe_list = []
    for i in data_positive["sequence"].unique():
        if len(i) >= 1:
            row_data = []
            row_data.append(i)
            row_data.append(data_positive[data_positive["sequence"]==i]["origin"].mean())
            row_data.append(1)
            dataframe_list.append(row_data)

    data_negative = pd.read_csv(negative_file_path,encoding="utf8")
    data_negative = data_negative[~data_negative["Sequence"].str.contains("B|X|Z|O|U")]
    data_negative.reset_index(drop=True, inplace=True)

    column_index = ["sequence","MIC","type"]
    for i in data_negative.index:
        if len(data_negative.loc[i,"Sequence"])>=1:
            row_data = []
            row_data.append(data_negative.loc[i,"Sequence"])
            row_data.append(8096)
            row_data.append(0)
            dataframe_list.append(row_data)
    data = pd.DataFrame(data=dataframe_list, columns=column_index)
    data = data.sample(frac=1)
    data.reset_index(drop=True, inplace=True)
    data.to_csv(output_path)

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

if __name__ == "__main__":
    generate_classifier_data()
    print("duplicate_mean.py")