#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:27:01 2023

@author: cyberguli
"""

from torch import nn
from models.basic_layers.unpool import UnPool
import torch
from torch_geometric.nn import ChebConv
class CTU(nn.Module):
    def __init__(self,in_channels,out_channels,l,adj_in,adj_out,drop_prob):
        super().__init__()
        self.l=l
        self.adj_in=adj_in
        self.adj_out=adj_out
        self.conv=ChebConv(in_channels,out_channels,3)
        self.pool=UnPool(l,self.adj_out)
        self.relu=nn.ReLU()
        self.batch=nn.BatchNorm1d(out_channels*(torch.amax(self.adj_out)+1))
        self.dropout=nn.Dropout(drop_prob)

    def forward(self,x):
        x=self.conv(x,self.adj_in)
        x=self.pool(x)
        tmp=x.shape
        x=x.reshape(x.shape[0],-1)
        x=self.relu(x)
        x=self.batch(x)
        x=self.dropout(x)
        return x.reshape(tmp)



