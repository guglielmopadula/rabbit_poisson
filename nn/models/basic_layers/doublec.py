#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:27:01 2023

@author: cyberguli
"""

from torch import nn
import torch

def lin_solve(A,b):
    return torch.bmm(torch.bmm(torch.transpose(A,1,2),torch.inverse(torch.bmm(A,torch.transpose(A,1,2)))),b)

class Barycenter(nn.Module):
    def __init__(self,batch_size,barycenter):
        super().__init__()
        self.barycenter=barycenter
        self.batch_size=batch_size

    def forward(self,x):
        x=x.reshape(x.shape[0],-1,3)
        return x-torch.mean(x,axis=1).unsqueeze(1).repeat(1,x.shape[1],1)+self.barycenter.unsqueeze(0).unsqueeze(0).repeat(x.shape[0],x.shape[1],1)

