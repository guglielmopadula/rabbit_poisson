#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:27:01 2023

@author: cyberguli
"""

from torch import nn
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist

class BR(PyroModule):
    def __init__(self,in_features,out_features):
        super().__init__()
        self.lin=PyroModule[nn.Linear](in_features, out_features)
        self.lin.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
        self.lin.bias = PyroSample(dist.Normal(0., 1.).expand([out_features]).to_event(1))
        self.relu=nn.ReLU()
    
    def forward(self,x):
        return self.relu(self.lin(x))


