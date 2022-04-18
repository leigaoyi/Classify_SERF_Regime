# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 14:49:40 2022

@author: leisir
"""

import torch
import torch.nn as nn



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
  
model = ANet()      
x = torch.randn((1, 1, 200))
y_pred = model(x)
print(y_pred.shape)
      