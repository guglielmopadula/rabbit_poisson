#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:27:01 2023

@author: cyberguli
"""

from torch import nn


class Pool(nn.Module):
    def __init__(self,l):
        super().__init__()
        self.l=l
    def forward(self,x):
        return x[:,self.l,:]



