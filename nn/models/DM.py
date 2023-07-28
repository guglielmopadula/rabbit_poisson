from pytorch_lightning import LightningModule
from torch import nn
import torch
import torch.autograd as autograd
import numpy as np
from models.basic_layers.lbr import LBR 


class HiddenNN(nn.Module):
    def __init__(self, in_dim,out_dim,drop_prob):
        super().__init__()
        self.fc1_interior = LBR(in_dim,in_dim,drop_prob)
        self.fc2_interior = LBR(in_dim,in_dim,drop_prob)
        self.fc3_interior = LBR(in_dim,out_dim,drop_prob)
        self.fc4_interior = LBR(out_dim,out_dim,drop_prob)
        self.fc5_interior = LBR(out_dim,out_dim,drop_prob)
        self.fc6_interior = LBR(out_dim,out_dim,drop_prob)
        self.fc7_interior = nn.Linear(out_dim,out_dim)
        
    def forward(self,x):
        x_hat=self.fc7_interior(self.fc6_interior(self.fc5_interior(self.fc4_interior(self.fc3_interior(self.fc2_interior(self.fc1_interior(x)))))))
        return x_hat



class DM(LightningModule):

    
    def __init__(self,latent_dim,drop_prob,decoder,hidden_dim: int= 500,T=10,beta_min=0.002,beta_max=0.2,**kwargs):
        super().__init__()
        self.drop_prob=drop_prob
        self.latent_dim=latent_dim
        self.hidden_dim=hidden_dim
        self.hidden_nn = HiddenNN(in_dim=self.latent_dim,out_dim=self.latent_dim,drop_prob=self.drop_prob)
        self.decoder=decoder
        self.T=T
        self.beta_min=beta_min
        self.beta_max=beta_max
        self.train_losses=[]
        self.eval_losses=[]

    def beta(self, t):
        return self.beta_min + (t / self.T) * (self.beta_max - self.beta_min)

    def alpha(self, t):
        return 1 - self.beta(t)

    def bar_alpha(self,t):
        return torch.prod(self.alpha(torch.arange(t)))
    
    def sigma(self,t):
        return self.beta(t)

    def training_step(self, batch, batch_idx):
        x=batch
        t=int(torch.floor(torch.rand(1)*self.T)+1)
        eps=torch.randn(x.shape).to(x.device)
        input=torch.sqrt(self.bar_alpha(t)).to(x.device)*batch+(torch.sqrt(1-self.bar_alpha(t))*eps).to(x.device)
        loss=torch.linalg.norm(eps-self.hidden_nn(input))
        self.log("train_ebm_loss", loss)
        self.train_losses.append(loss.item())
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        x=batch
        t=int(torch.floor(torch.rand(1)*self.T)+1)
        eps=torch.randn(x.shape).to(x.device)
        input=torch.sqrt(self.bar_alpha(t)).to(x.device)*batch+(torch.sqrt(1-self.bar_alpha(t))*eps).to(x.device)
        loss=torch.linalg.norm(eps-self.hidden_nn(input))
        self.eval_losses.append(loss.item())
        return loss

    def test_step(self, batch, batch_idx):
        x=batch
        t=int(torch.floor(torch.rand(1)*self.T)+1)
        eps=torch.randn(x.shape).to(x.device)
        input=torch.sqrt(self.bar_alpha(t)).to(x.device)*batch+(torch.sqrt(1-self.bar_alpha(t))*eps).to(x.device)
        loss=torch.linalg.norm(eps-self.hidden_nn(input))
        return loss
    

    def get_latent(self,data):
        return self.encoder.forward(data)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)#0.001 k=1
        return {"optimizer": optimizer}

    def sample_mesh(self,mean=None,var=None):
        device=self.decoder.pca._V.device
        self=self.to(device)
        if mean==None:
            mean=torch.zeros(1,self.latent_dim)

        if var==None:
            var=torch.ones(1,self.latent_dim)
        x=torch.randn(var.shape[0],self.latent_dim)
        x=x.to(device)

        for t in reversed(range(1,self.T+1)):
            z=torch.randn(var.shape[0],self.latent_dim)
            z=z.to(device)
            x=1/torch.sqrt(self.bar_alpha(t))*(x-(1-self.alpha(t))/(1-self.bar_alpha(t))*self.hidden_nn(x))+self.sigma(t)*z
        tmp=self.decoder(x)
        return tmp,x
     
