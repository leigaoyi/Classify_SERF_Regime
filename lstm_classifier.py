# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 11:21:10 2022

@author: leisir
"""

# import wavencoder models and trainer
from wavencoder.models import Wav2Vec, LSTM_Attn_Classifier
from wavencoder.trainer import train, test_evaluate_classifier, test_predict_classifier
from wavencoder.models import CNN1d
# import torch modules and torchaudio for data
#import torchaudio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset

import random
#import IPython
from tqdm import tqdm
import numpy as np
#import matplotlib.pyplot as plt 
import yaml

import pickle

## 导入数据集，自定义
#from read_excel import labels_shuffle, data_shuffle

with open('./ckpt/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)

##准备数据集函数，列出模型。
data_shuffle = np.load(config['dataset']['data'])
labels_shuffle = np.load('./dataset/label.npy')

class AborptionDataset(Dataset):
      def __init__(self, data, label_dict):
            self.dataset = data
            self.label_dict = label_dict
      
      def __len__(self):
            return len(self.dataset)

      def __getitem__(self, idx):
            wave_idx = self.dataset[idx, :]
            wave_idx = np.reshape(wave_idx, (1, -1))
            label_idx = self.label_dict[idx][0]
            label_idx = np.reshape(label_idx, (1))
            return wave_idx, label_idx
      
absorption_data = AborptionDataset(data_shuffle, labels_shuffle)      
# 预计，(352,200) data. (352, 1) label

train_len = int(len(absorption_data)*0.8)      
val_len = len(absorption_data) - train_len

train_ds, val_ds = random_split(absorption_data, [train_len, val_len])

trainloader = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=True)
valloader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

batch = next(iter(trainloader))
x, y = batch

print(x.shape, y.shape)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class_num = 2
model = nn.Sequential(
    #Wav2Vec(pretrained=True),
    nn.Conv1d(1, 4, 1),
    nn.ReLU(),
    LSTM_Attn_Classifier(4, 8, class_num)
)
print(model)

class ANet(nn.Module):
      '''
      in_x : [Batch, 1, 200]
      y " [Batch, 2]"
      '''
      def __init__(self, num_classes=2):
          super(ANet, self).__init__()
          self.features = nn.Sequential(
              nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3),
              nn.ReLU(inplace=True),
              nn.Conv1d(in_channels=4, out_channels=2, kernel_size=3),
              nn.ReLU(inplace=True)
          )
          self.classifier = nn.Linear(392, num_classes)
      
      def forward(self, x):
          x = self.features(x)
          x = x.view(x.size(0), -1)
          x = self.classifier(x)
          return x
#model = ANet()


#optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-1, step_size_up=10)
model, train_dict = train(model, trainloader, valloader, n_epochs=config['epoch'],\
                          optimizer=optimizer)
file = open('./ckpt/train_dict.pickle', 'wb')
pickle.dump(train_dict)
file.close()
      
