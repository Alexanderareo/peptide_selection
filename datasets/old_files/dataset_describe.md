# 数据集描述

**all_data_with_negative_8086.csv**
> 1. 10398条，正样例为经过完整筛选的2300条多肽，负样例为筛去特殊氨基酸的8100条多肽组成。  
> 2. 所有的MIC值均为原始值，其中正样例并未筛选仅针对金色葡萄球菌的数据，简单的针对各菌种的抗菌原始值取了均值，负样例统一取值为8096  

**all_data_with_negative_features.csv**
> 在第一个数据集的基础上，根据之前的之前的多肽features的计算公式，把10398条都计算了一遍参数，由于我们要找的是6肽，所以计算中把所有长度小于6的肽都给筛去了  

**all_data_with_negative_labels.csv**
> 在第一个数据集的基础上，给正样例和负样例添加了标签，用于分类任务。

**filtered_negative.csv**
> 负样例的初始数据，共8600条，其中有一部分不符合要求，接下来的任务按需剔除。  

**grampa_all_data_unique_with_mean.csv**
> grampa数据集中所有的独特序列，6760条，这里对他们的不同菌种的抗菌性取均值并还原成原始值  

**grampa_filtered_with_origin_result**
> 在grampa数据集上经过筛选的，符合生物那边要求的14861条数据，经过去重后即为正样例  
> 将原始的log10还原为原始值

**grampa_filtered.csv**
> 在grampa数据集上经过筛选的，符合生物那边要求的14861条数据，经过去重后即为正样例  
> 未还原初始值

**test.csv**
> 实验性地将grampa中地6760条不重复的肽取均值后作为正样例(不经过筛选)，加入之前的负样例，形成一个14807条的数据集，是能从已有的数据中汇总出的条数最多的数据集