import sys

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import random

from Bio import SeqIO

from settings import MAX_SEQUENCE_LENGTH, character_to_index, CHARACTER_DICT, max_mic_buffer, MAX_MIC
import sys

"""
主要是针对生成负样例的
"""
def get_bacterium_df(bacterium, df):
    # 根据 bacterium 筛选菌种
    bacterium_df = df.loc[(df.bacterium.str.contains(bacterium))].groupby(['sequence', 'bacterium'])
    return bacterium_df.mean().reset_index().dropna()

def sequence_to_vector(sequence):
    # onehot编码    vector (47,21)
    vector = np.zeros([MAX_SEQUENCE_LENGTH, len(character_to_index) + 1])
    for i, character in enumerate(sequence[:MAX_SEQUENCE_LENGTH]):
        vector[i][character_to_index[character]] = 1
    return vector

def value2log(value):
    value = np.log10(value)
    return value

def generate_random_sequence(min_length=5, max_length=MAX_SEQUENCE_LENGTH, fixed_length=None):
    # 生成随机序列
    # return 随机长度的序列 ['C', 'Y', 'N', 'M', 'T', 'K', 'I', 'Q', 'S', 'E']
    # 从分布来讲，抗菌肽的数量是远远小于不抗菌的，这样一来也没有问题
    if fixed_length:
        length = fixed_length
    else:
        length = random.choice(range(min_length, max_length))
    sequence = [random.choice(list(CHARACTER_DICT)) for _ in range(length)]

    return sequence

def add_random_negative_examples(vectors, labels, negatives_ratio):
    # 生成随机负样例序列
    # vector为正样例 
    if negatives_ratio == 0:
        return vectors, labels
    num_negative_vectors = int(negatives_ratio * len(vectors))
    negative_vectors = np.array(
        [sequence_to_vector(generate_random_sequence()) for _ in range(num_negative_vectors)]
    ) 
    vectors = np.concatenate((vectors, negative_vectors))
    negative_labels = np.full(num_negative_vectors, MAX_MIC) # 一次性赋值完所有负样本
    labels = np.concatenate((labels, negative_labels))
    # print(vectors[0],labels[0])
    return vectors, labels

def load_uniprot_negatives(count):
    """
    加载uniprot的负样例，剔除含有'C'的负样例    
    sequence为其中的片段
    返回的sequence数目为count
    """  
    uniprot_file = 'data_cnn/Fasta_files/Uniprot_negatives.txt'
    fasta = SeqIO.parse(uniprot_file, 'fasta')
    fasta_sequences = [str(f.seq) for f in fasta]
    negatives = []
    for seq in fasta_sequences:
        if 'C' in seq:
            continue
        start = random.randint(0,len(seq)-MAX_SEQUENCE_LENGTH)
        negatives.append(seq[start:(start+MAX_SEQUENCE_LENGTH)])
        if len(negatives) >= count:
            return negatives
    return negatives

def uniprot_precision(model):
    negatives = load_uniprot_negatives(1000) # 每次调1000个负样例
    vectors = []
    for seq in negatives:
        try:
            vectors.append(sequence_to_vector(seq)) # 转化为矩阵
        except KeyError:
            continue
    preds = model.predict(np.array(vectors)) # 用模型预测
    # false_positives = len([p for p in preds if p < MAX_MIC - max_mic_buffer]) # 误差的数目
    false_positives = len([p for p in preds if p < 3.5])
    return 1 - false_positives / len(negatives)  # 负样例的预测正确率
