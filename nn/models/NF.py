from pytorch_lightning import LightningModule
from torch import nn
from models.basic_layers.ld import Latent_Discriminator_base
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

class NVP(nn.Module):
    def __init__(self, latent_dim,drop_prob):
        super().__init__()
        self.latent_dim=latent_dim
        perm=torch.randperm(latent_dim)
        self.pres=perm[:len(perm)//2]
        self.notpres=perm[len(perm)//2:]
        self.s=HiddenNN(len(self.pres),len(self.notpres),drop_prob)
        self.m=HiddenNN(len(self.pres),len(self.notpres),drop_prob)    
    
    def forward(self,x):
        x1=x.reshape(-1,self.latent_dim)[:,self.pres]
        x2=x.reshape(-1,self.latent_dim)[:,self.notpres]
        z1=x1
        z2=torch.exp(self.s(x1))*x2+self.m(x1)
        z=torch.zeros(z1.shape[0],self.latent_dim).to(z2.device)
        z[:,self.pres]=z1
        z[:,self.notpres]=z2
        return z,self.s(x1).sum(1)
    
    def my_backward(self,z):
        z1=z.reshape(-1,self.latent_dim)[:,self.pres]
        z2=z.reshape(-1,self.latent_dim)[:,self.notpres]
        x1=z1
        x2=torch.exp(-self.s(z1))*(z2-self.m(z1))
        x=torch.zeros(z1.shape[0],self.latent_dim).to(x1.device)
        x[:,self.pres]=x1
        x[:,self.notpres]=x2
        return x,-self.s(z1).sum(1)
    
class NF(LightningModule):
    def __init__(self, latent_dim,drop_prob,decoder):
        super().__init__()
        self.modulelist=nn.ModuleList([NVP(latent_dim,drop_prob) for i in range(5)])
        self.decoder=decoder
        self.dist=torch.distributions.Normal(loc=0.,scale=1.)
        self.automatic_optimization=False
        self.latent_dim=latent_dim
        self.train_losses=[]
        self.eval_losses=[]

    def forward(self,x):
        det=0
        for flow in self.modulelist:
            x,det_tmp=flow(x)
            det=det+det_tmp

        return x,det
    
    def my_backward(self,z):
        det=0
        for flow in reversed(self.modulelist):
            z,det_tmp=flow.my_backward(z)
            det=det+det_tmp
        return z,det


    def training_step(self, batch, batch_idx):
        opt=self.optimizers()
        x=batch
        z,lk=self.forward(x)
        lnorm=self.dist.log_prob(z).sum(1)
        loss=-torch.mean(lk+lnorm)
        self.manual_backward(loss)
        self.clip_gradients(opt,0.1)
        opt.step()
        opt.zero_grad()
        self.train_losses.append(loss.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        x=batch
        z,lk=self.forward(x)
        lnorm=self.dist.log_prob(z).sum(1)
        loss=-torch.mean(lk+lnorm)
        self.eval_losses.append(loss.item())
        return loss
    
    def test_step(self, batch, batch_idx):
        x=batch
        z,lk=self.forward(x)
        lnorm=self.dist.log_prob(z).sum(1)
        loss=-torch.mean(lk+lnorm)
        return loss


    def sample_mesh(self,mean=None,var=None):
        device=self.decoder.pca._V.device
        self=self.to(device)
        if mean==None:
            mean=torch.zeros(1,self.latent_dim)

        if var==None:
            var=torch.ones(1,self.latent_dim)
        z=torch.randn(var.shape[0],self.latent_dim)
        z=z.to(device)
        z,_=self.my_backward(z)
        tmp=self.decoder(z)
        return tmp,z


    def test_step(self, batch, batch_idx):
        return 0
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001)#0.0001 k=1
        return {"optimizer": optimizer}
