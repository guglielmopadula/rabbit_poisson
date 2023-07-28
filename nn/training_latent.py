#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:33:45 2023

@author: cyberguli
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
from datawrapper.encoded_data import EncodedData
import sys
from models.DM import DM
from models.NF import NF
from models.EBM import EBM
import copy
import torch
import numpy as np
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
NUM_TRAIN_SAMPLES=400#400
NUM_TEST_SAMPLES=200#200
BATCH_SIZE = 200
LATENT_DIM=15#3
MAX_EPOCHS={"AE":500,"BEGAN":500,"AAE":500,"VAE":500}
SMOOTHING_DEGREE=1
DROP_PROB=0.1

encoded_data_test=torch.load("./data/encoded_data_test.pt")
encoded_data_train=torch.load("./data/encoded_data_train.pt")

data=EncodedData(num_workers=NUM_WORKERS,batch_size=BATCH_SIZE,data_train=encoded_data_train,data_test=encoded_data_test,use_cuda=use_cuda,latent_dim=encoded_data_train.shape[1])

                               

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


d={
    DM: "DM",
  EBM: "EBM",
  NF: "NF", 
}
decoder=torch.load("./nn/saved_models/AE.pt").decoder.decoder_base
if __name__ == "__main__":
    name=sys.argv[1]
    wrapper=list(d.keys())[list(d.values()).index(name)]
    torch.manual_seed(100)
    np.random.seed(100)
    if data.use_cuda:
        trainer = Trainer(accelerator='gpu', devices=AVAIL_GPUS,max_epochs=500,log_every_n_steps=1,
                                plugins=[DisabledSLURMEnvironment(auto_requeue=False)],
                                )
    else:
        trainer=Trainer(max_epochs=MAX_EPOCHS[name],log_every_n_steps=1,
                            plugins=[DisabledSLURMEnvironment(auto_requeue=False)],accelerator="cpu"
                            )

    model=wrapper(latent_dim=LATENT_DIM,drop_prob=DROP_PROB,decoder=decoder)
    print("Training of "+name+ "has started")
    trainer.fit(model, data)
    np.save("./nn/saved_models/"+name+"_train_loss.npy",np.array(model.train_losses))
    np.save("./nn/saved_models/"+name+"_eval_loss.npy",np.array(model.eval_losses))

    torch.save(model,"./nn/saved_models/"+name+".pt")
    data=EncodedData(num_workers=NUM_WORKERS,batch_size=BATCH_SIZE,data_train=encoded_data_train,data_test=encoded_data_test,use_cuda=False,latent_dim=encoded_data_train.shape[1])
                               

    model=torch.load("./nn/saved_models/"+name+".pt",map_location="cpu")
    model=model.eval()  
    with torch.no_grad():
        print(custom_test(model,data))


    
    

    
    
    
    
    
    
    
    
    
    
    
