#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:33:45 2023

@author: cyberguli
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
from datawrapper.disc_data import DiscData
import sys
from models.BayesDiscriminator import Discriminator
import numpy as np
import torch
import dill
from pytorch_lightning import Trainer
torch.autograd.set_detect_anomaly(True)
from pytorch_lightning.plugins.environments import SLURMEnvironment
torch.set_float32_matmul_precision('high')
torch.use_deterministic_algorithms(True)
class DisabledSLURMEnvironment(SLURMEnvironment):
    def detect() -> bool:
        return False

    @staticmethod
    def _validate_srun_used() -> None:
        return

    @staticmethod
    def _validate_srun_variables() -> None:
        return


NUM_WORKERS = os.cpu_count()//2
use_cuda=True if torch.cuda.is_available() else False


use_cuda=True if torch.cuda.is_available() else False

AVAIL_GPUS=1 if use_cuda else 0

REDUCED_DIMENSION=36
NUM_TRAIN_SAMPLES=800#400
NUM_TEST_SAMPLES=400#200
BATCH_SIZE = 400
LATENT_DIM=15#3
MAX_EPOCHS={"AE":500,"BEGAN":500,"AAE":500,"VAE":500}
SMOOTHING_DEGREE=1
DROP_PROB=0.1



data=DiscData(batch_size=BATCH_SIZE,num_train=NUM_TRAIN_SAMPLES,
          num_test=NUM_TEST_SAMPLES,
          num_workers=NUM_WORKERS,
          dataset=torch.load("data/disc_data.pt"))

data_shape=torch.load("data/disc_data.pt")[:][0][0].reshape(-1).shape[0]

def custom_test(model,data):
    iterator=iter(data.test_dataloader())
    n_batches=data.num_test//data.batch_size
    tot_loss=0
    for i in range(n_batches):
        batch=next(iterator)
        loss=model.test_step(batch,0)
        tot_loss=tot_loss+loss
    tot_loss=tot_loss/n_batches
    return tot_loss


torch.manual_seed(100)
np.random.seed(100)
trainer = Trainer(devices=AVAIL_GPUS,max_epochs=500,log_every_n_steps=1,
                            plugins=[DisabledSLURMEnvironment(auto_requeue=False)],
                            )
model=Discriminator(data_shape,data_shape,data.batch_size)
model.train(data.data_train[:],500)
with open("nn/saved_models/bayesdisc.pt", 'wb') as out_strm: 
    dill.dump(model, out_strm)

model.eval()
y_fit=model.predict(data.data_train[:])
y_test=model.predict(data.data_test[:])
y_fit=np.array(model.predict(data.data_train[:]).detach()).reshape(-1)
y_pred=np.array(model.predict(data.data_test[:]).detach()).reshape(-1)
y_train=np.array(data.data_train[:][1].detach()).reshape(-1)
y_test=np.array(data.data_test[:][1].detach()).reshape(-1)
print(y_train.shape)
print(y_test.shape)
print(y_pred.shape)
print(y_fit.shape)
print(np.sum(np.abs(y_train-y_fit))/len(y_fit))
print(np.sum(np.abs(y_test-y_pred))/len(y_test))    

    
    
    
    
    
    
    
    
    
    
    
