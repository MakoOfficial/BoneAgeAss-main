import torch
import os
import numpy as np
import pandas as pd
import collections

# d2l里的获取标签
def read_csv_labels_noMale(fname):
    """读取标签，返回字典格式`"""
    with open(fname, 'r') as f:
        # Skip the file header line (column name) 跳过文件头
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    # return dict(((id, boneage) for id, boneage, male in tokens))
    return dict((id, boneage) for id, boneage, male in tokens)

labels = read_csv_labels_noMale('../../data/archive/valid-dataset.csv')
print(collections.Counter(labels.values()))
