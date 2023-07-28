from pytorch_lightning import LightningModule
from torch import nn
from models.basic_layers.br import BR
import itertools
from models.losses.losses import L2_loss,CE_loss
import torch
from models.basic_layers.PCA import PCA
from pyro.nn import PyroModule, PyroSample
import pyro
import pyro.distributions as dist
class Bayes_Discriminator_base(PyroModule):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.fc1_interior = BR(latent_dim,hidden_dim)
        self.fc2_interior = BR(hidden_dim,hidden_dim)
        self.fc3_interior = BR(hidden_dim,hidden_dim)
        self.fc4_interior = BR(hidden_dim,hidden_dim)
        self.fc5_interior = BR(hidden_dim,hidden_dim)
        self.fc6_interior = BR(hidden_dim,hidden_dim)
        self.fc7_interior = PyroModule[nn.Linear](hidden_dim,1)
        self.fc7_interior.weight = PyroSample(dist.Normal(0., 1.).expand([1, hidden_dim]).to_event(2))
        self.fc7_interior.bias = PyroSample(dist.Normal(0., 1.).expand([1]).to_event(1))
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,x,y=None):
        a=self.fc6_interior.lin(self.fc5_interior(self.fc4_interior(self.fc3_interior(self.fc2_interior(self.fc1_interior(x))))))
        prob=self.sigmoid(self.fc7_interior(self.fc6_interior(self.fc5_interior(self.fc4_interior(self.fc3_interior(self.fc2_interior(self.fc1_interior(x)))))))).squeeze()
        with pyro.plate("data", x.shape[0]):
            a=pyro.deterministic("latent_value",a)
            obs = pyro.sample("obs", dist.Bernoulli(prob), obs=y)