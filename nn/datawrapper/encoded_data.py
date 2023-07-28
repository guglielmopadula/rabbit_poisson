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



class EncodedData(LightningDataModule):
    
    def get_size(self):
        return self.latent_dim

    def __init__(
        self,batch_size,num_workers,data_train,data_test,latent_dim,use_cuda):
        super().__init__()
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.use_cuda=use_cuda
        self.num_workers = num_workers
        self.latent_dim=latent_dim
        self.data_train=data_train
        self.data_test=data_test
        self.num_train=len(data_train)
        self.num_test=len(data_test)

    
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
    
    def val_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )