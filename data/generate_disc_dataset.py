#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 00:09:55 2023

@author: cyberguli
"""
import torch
import numpy as np
from tqdm import trange
from sklearn.decomposition import PCA
REDUCED_DIMENSION=36
true_data=np.load("data/data.npy")
NUM_SAMPLES=len(true_data)
false_data=np.load("data/neg_data.npy")
n_true=len(true_data)
n_false=len(false_data)
y_true=np.ones(n_true)
y_false=np.ones(n_false)
x=np.concatenate((true_data,false_data))
y=np.concatenate((y_true,y_false))
y=torch.tensor(y,dtype=torch.float32)
x=torch.tensor(x,dtype=torch.float32)
data=torch.utils.data.TensorDataset(x,y)
torch.save(data,"data/disc_data.pt")