#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 00:09:55 2023

@author: cyberguli
"""
from datawrapper.data import Data
from datawrapper.encoded_data import EncodedData
from models.AE import AE
from datawrapper.data import Data
import torch
import numpy as np
NUM_WORKERS = 0
use_cuda=True if torch.cuda.is_available() else False
AVAIL_GPUS=1 if torch.cuda.is_available() else 0
REDUCED_DIMENSION=36
NUM_TRAIN_SAMPLES=400
NUM_TEST_SAMPLES=200
NUM_VAL_SAMPLES=0
BATCH_SIZE = 200
LATENT_DIM=15#3


data=Data(batch_size=BATCH_SIZE,num_train=NUM_TRAIN_SAMPLES,
          num_test=NUM_TEST_SAMPLES,
          num_workers=NUM_WORKERS,
          reduced_dimension=REDUCED_DIMENSION, 
          data=np.load("data/data.npy")[:600,],
          use_cuda=True)
                               

AE=torch.load("./nn/saved_models/AE.pt",map_location=torch.device('cpu'))
data.data_train=data.data_train[:].reshape(NUM_TRAIN_SAMPLES//BATCH_SIZE,BATCH_SIZE,-1,3)

data.data_test=data.data_test[:].reshape(NUM_TEST_SAMPLES//BATCH_SIZE,BATCH_SIZE,-1,3)

encoded_data_train=torch.zeros(NUM_TRAIN_SAMPLES,LATENT_DIM).reshape(NUM_TRAIN_SAMPLES//BATCH_SIZE,BATCH_SIZE,LATENT_DIM)
encoded_data_test=torch.zeros(NUM_TEST_SAMPLES,LATENT_DIM).reshape(NUM_TEST_SAMPLES//BATCH_SIZE,BATCH_SIZE,LATENT_DIM)

AE.encoder(data.data_train[0])

for i in range(NUM_TRAIN_SAMPLES//BATCH_SIZE):
    tmp=AE.encoder(data.data_train[i])
    encoded_data_train[i]=tmp.reshape(BATCH_SIZE,-1)
for i in range(NUM_TEST_SAMPLES//BATCH_SIZE):
    encoded_data_test[i]=AE.encoder(data.data_test[i])


torch.save(encoded_data_test.detach().reshape(NUM_TEST_SAMPLES,-1),"./data/encoded_data_test.pt")
torch.save(encoded_data_train.detach().reshape(NUM_TRAIN_SAMPLES,-1),"./data/encoded_data_train.pt")

encoded_data_wrapper=EncodedData(num_workers=data.num_workers,batch_size=data.batch_size,data_train=encoded_data_test,data_test=encoded_data_test,use_cuda=use_cuda,latent_dim=encoded_data_train.shape[1])
