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
from torch.utils.data import random_split



class DiscData(LightningDataModule):
    

    def __init__(
        self,batch_size,num_workers,dataset,num_train,num_test):
        super().__init__()
        self.batch_size=batch_size
        self.num_train=num_train
        self.num_workers = num_workers
        self.num_test=num_test
        self.num_samples=self.num_test+self.num_train
        self.data=dataset
        self.data_shape=self.data[:][0][0].shape
        self.data_train,self.data_test = random_split(self.data, [self.num_train,self.num_test],generator=torch.Generator().manual_seed(100))    

    
    def prepare_data(self):
        pass


    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers)        
    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )