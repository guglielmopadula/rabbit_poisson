from pytorch_lightning import LightningModule
from torch import nn
from models.basic_layers.ld import Latent_Discriminator_base
import torch
import torch.autograd as autograd
import numpy as np
class Discriminator(nn.Module):
    def __init__(self, latent_dim,hidden_dim,drop_prob):
        super().__init__()
        self.discriminator=Latent_Discriminator_base(latent_dim=latent_dim,hidden_dim=hidden_dim, drop_prob=drop_prob)
            
    def forward(self,x):
        x_hat=self.discriminator(x)
        return x_hat

def sample_langevin(x, model, stepsize, n_steps):
    l_samples = []
    l_dynamics = []
    x.requires_grad = True
    noise_scale = np.sqrt(stepsize * 2)
    for _ in range(n_steps):
        l_samples.append(x.detach())
        noise = torch.randn_like(x) * noise_scale
        out = model(x)
        if out.requires_grad==False:
            out.requires_grad=True
        grad = autograd.grad(out.sum(), x, only_inputs=True)[0]
        dynamics = stepsize * grad + noise
        x = x + dynamics
        l_samples.append(x.detach())
        l_dynamics.append(dynamics.detach())
    return l_samples[-1]


class EBM(LightningModule):
    
    class Discriminator(nn.Module):
        def __init__(self, latent_dim,hidden_dim,drop_prob):
            super().__init__()
            self.discriminator=Latent_Discriminator_base(latent_dim=latent_dim,hidden_dim=hidden_dim, drop_prob=drop_prob)
                
        def forward(self,x):
            x_hat=self.discriminator(x)
            return x_hat
    
    def __init__(self,latent_dim,drop_prob,decoder,hidden_dim: int= 500,**kwargs):
        super().__init__()
        self.drop_prob=drop_prob
        self.latent_dim=latent_dim
        self.hidden_dim=hidden_dim
        self.discriminator = self.Discriminator(latent_dim=self.latent_dim,hidden_dim=self.hidden_dim,drop_prob=self.drop_prob)
        self.decoder=decoder
        self.train_losses=[]
        self.eval_losses=[]

    def training_step(self, batch, batch_idx):
        pos_x=batch
        neg_x = torch.randn_like(pos_x)
        neg_x = sample_langevin(neg_x, self.discriminator, 0.01, 50)
        pos_out = self.discriminator(pos_x)
        neg_out = self.discriminator(neg_x)
        loss = (pos_out - neg_out) + 10 * (pos_out ** 2 + neg_out ** 2)
        loss = loss.mean()
        self.log("train_ebm_loss", loss)
        self.train_losses.append(loss.item())
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        pos_x=batch
        neg_x = torch.randn_like(pos_x)
        neg_x = sample_langevin(neg_x, self.discriminator, 0.01, 50)
        pos_out = self.discriminator(pos_x)
        neg_out = self.discriminator(neg_x)
        loss = (pos_out - neg_out) + 10 * (pos_out ** 2 + neg_out ** 2)
        loss = loss.mean()
        self.log("train_ebm_loss", loss)
        self.eval_losses.append(loss.item())
        torch.set_grad_enabled(had_gradients_enabled)
        return loss
    
    def test_step(self, batch, batch_idx):
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        pos_x=batch
        neg_x = torch.randn_like(pos_x)
        neg_x = sample_langevin(neg_x, self.discriminator, 0.01, 50)
        pos_out = self.discriminator(pos_x)
        neg_out = self.discriminator(neg_x)
        loss = (pos_out - neg_out) + 10 * (pos_out ** 2 + neg_out ** 2)
        loss = loss.mean()
        self.log("train_ebm_loss", loss)
        self.eval_losses.append(loss.item())
        torch.set_grad_enabled(had_gradients_enabled)
        return loss

    def get_latent(self,data):
        return self.encoder.forward(data)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.00009)#0.0001 k=1
        return {"optimizer": optimizer}

    def sample_mesh(self,mean=None,var=None):
        device=self.decoder.pca._V.device
        self=self.to(device)
        if mean==None:
            mean=torch.zeros(1,self.latent_dim)

        if var==None:
            var=torch.ones(1,self.latent_dim)
        z=torch.randn(var.shape[0],self.latent_dim)
        z=z.to(device)
        z=sample_langevin(z, self.discriminator, 0.01, 50)
        tmp=self.decoder(z)
        return tmp,z
     
