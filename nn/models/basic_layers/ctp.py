#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:27:01 2023

@author: cyberguli
"""

from torch import nn
from models.basic_layers.pool import Pool
import torch
from torch_geometric.nn import ChebConv, BatchNorm
class CTP(nn.Module):
    def __init__(self,in_channels,out_channels,l,adj,drop_prob):
        super().__init__()
        self.l=l
        self.conv=ChebConv(in_channels,out_channels,3)
        self.pool=Pool(l)
        self.adj=adj
        self.relu=nn.ReLU()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.batch=BatchNorm(out_channels*(torch.amax(adj)+1))
        self.dropout=nn.Dropout(drop_prob)

    def forward(self,x):
        x=self.pool(x)
        x=self.conv(x,self.adj)
        x=self.relu(x)
        tmp=x.shape
        x=x.reshape(x.shape[0],-1)
        x=self.batch(x)
        x=self.dropout(x)
        return x.reshape(tmp)



