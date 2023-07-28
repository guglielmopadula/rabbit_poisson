#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:27:01 2023

@author: cyberguli
"""

from torch import nn
import torch

class UnPool(nn.Module):
    def __init__(self,l,adj_out):
        super().__init__()
        self.l=l
        self.adj_out=adj_out
    def forward(self,x):
        y=torch.zeros(x.shape[0],torch.amax(self.adj_out)+1,x.shape[2],device=x.device)
        y[:,self.l,:]=x
        return y



