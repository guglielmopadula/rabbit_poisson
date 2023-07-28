from pytorch_lightning import LightningModule
from torch import nn
from models.basic_layers.bd import Bayes_Discriminator_base
import itertools
import torch
from models.basic_layers.PCA import PCA
from pyro.infer.autoguide import AutoDiagonalNormal,AutoDelta
from pyro.infer import SVI, Trace_ELBO, Predictive
import pyro
from tqdm import trange
class Discriminator():
    
    def __init__(self,data_shape,latent_dim,batch_size,hidden_dim: int= 500):
        super().__init__()
        self.latent_dim=latent_dim
        self.batch_size=batch_size
        self.hidden_dim=hidden_dim
        self.data_shape = data_shape
        self.discriminator=Bayes_Discriminator_base(latent_dim=latent_dim,hidden_dim=hidden_dim)
        self.guide = AutoDelta(self.discriminator)
        self.adam = pyro.optim.Adam({"lr": 1e-3})
        self.svi = SVI(self.discriminator, self.guide, self.adam, loss=Trace_ELBO())

    def train(self, data,epochs):
        x,y=data
        y=y.reshape(-1)
        x=x.reshape(x.shape[0],-1)
        for epoch in trange(epochs):
            loss = self.svi.step(x, y)
        return loss
    
    def eval(self):
        self.predictive=Predictive(self.discriminator, guide=self.guide, num_samples=1)

    def predict(self,batch):
        x,y=batch
        preds = self.predictive(x)['obs'].reshape(-1)
        return preds

    def compute_latent(self,batch):
        x,y=batch
        preds = self.predictive(x)['latent_value'].reshape(x.shape[0],-1)
        return preds
