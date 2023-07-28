#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:28:55 2023

@author: cyberguli
"""
import torch
import meshio
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import copy
from scipy.sparse import coo_array
from models.basic_layers.PCA import PCA
from torch.utils.data import random_split
from tqdm import trange
torch.set_default_dtype(torch.float32)



class Data(LightningDataModule):
    def get_size(self):
        return ((1,self.n_points,3))
    
    def get_reduced_size(self):
        return self.reduced_dimension

    def __init__(
        self,batch_size,num_workers,num_train,num_test,reduced_dimension,use_cuda,data):
        super().__init__()
        self.l=[]
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.use_cuda=use_cuda
        self.num_train=num_train
        self.num_workers = num_workers
        self.num_test=num_test
        self.reduced_dimension=reduced_dimension
        self.num_samples=self.num_test+self.num_train
        #barycenter=np.mean(data[0].reshape(-1,3),axis=0)
        self.barycenter=torch.tensor(0)
        self.n_points=data.shape[0]
        self.data=torch.tensor(data,dtype=torch.float32)
        self.data=self.data[:self.num_samples]
        self.pca=PCA(self.reduced_dimension)
        if use_cuda:
            self.pca.fit(self.data.reshape(self.num_samples,-1).cuda())
            self.barycenter=self.barycenter.cuda()
        else:
            self.pca.fit(self.data.reshape(self.num_samples,-1))

        self.data_train,self.data_test = random_split(self.data, [self.num_train,self.num_test],generator=torch.Generator().manual_seed(100))    

    
    def prepare_data(self):
        pass


    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    
    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
