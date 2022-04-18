# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:02:13 2022

@author: leisir
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_serf = pd.read_excel('data/serf/serf_152.xlsx', header=None)
df_nonserf1 = pd.read_excel('data/non_serf/non_serf_150.xlsx', header=None)
df_nonserf2 = pd.read_excel('data/non_serf/non_serf_140.xlsx', header=None)

data_serf = df_serf.values
data_nonserf1 = np.asarray(df_nonserf1.values)
data_nonserf2 = np.asarray(df_nonserf2.values)
data_nonserf = np.vstack((data_nonserf1, data_nonserf2))
print(np.mean(data_nonserf2))
# 数据归一化
#data_serf = (data_serf - data_serf.min())/(data_serf.max()-data_serf.min()+1e-8) - 0.5
#data_nonserf = (data_nonserf - data_nonserf.min())/(data_nonserf.max() - data_nonserf.min()+1e-8) - 0.5
data_serf = (data_serf - np.mean(data_serf))/np.std(data_serf)
data_nonserf = (data_nonserf - np.mean(data_nonserf))/np.std(data_nonserf)

print(data_nonserf)

label_serf = np.ones((data_serf.shape[0], 1))
label_nonserf = np.zeros((data_nonserf.shape[0],1))

# 打乱排布
labels = np.vstack((label_serf, label_nonserf))
data = np.vstack((data_serf, data_nonserf))
idx = np.asarray([i for i in range(data.shape[0])])
np.random.shuffle(idx)

labels_shuffle = labels[idx, :]
data_shuffle = data[idx, :]


np.save('data.npy', data_shuffle)
np.save('label.npy', labels_shuffle)
