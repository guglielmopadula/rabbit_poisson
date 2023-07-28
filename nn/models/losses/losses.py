
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
import gpytorch
import torch
def L2_loss(x_hat, x):
    loss=F.mse_loss(x.reshape(-1), x_hat.reshape(-1), reduction="none")
    loss=loss.mean()
    return loss

def get_kernel():
    s=gpytorch.kernels.RBFKernel()
    l=[1,2,5,10,20]
    for i in l:
        tmp=gpytorch.kernels.RBFKernel()
        tmp.lengthscale=i
        s=s+tmp
    return s

def mmd(X,Y):
    s=get_kernel()
    X=X.reshape(X.shape[0],-1)
    Y=Y.reshape(Y.shape[0],-1)
    return np.mean(s(torch.tensor(X),torch.tensor(X)).to_dense().detach().numpy())+np.mean(s(torch.tensor(Y),torch.tensor(Y)).to_dense().detach().numpy())-2*np.mean(s(torch.tensor(X),torch.tensor(Y)).to_dense().detach().numpy())

def relmmd(X,Y):
    s=get_kernel()
    X=X.reshape(X.shape[0],-1)
    Y=Y.reshape(Y.shape[0],-1)
    return (np.mean(s(torch.tensor(X),torch.tensor(X)).to_dense().detach().numpy())+np.mean(s(torch.tensor(Y),torch.tensor(Y)).to_dense().detach().numpy())-2*np.mean(s(torch.tensor(X),torch.tensor(Y)).to_dense().detach().numpy()))/(np.mean(s(torch.tensor(Y),torch.tensor(Y)).to_dense().detach().numpy()))


'''  
def mmd(X,Y):
    X=X.reshape(X.shape[0],-1)
    Y=Y.reshape(Y.shape[0],-1)
    return np.sqrt((1/(len(X)**2)*np.sum(pairwise_kernels(X, X, metric='laplacian'))+1/(len(Y)**2)*np.sum(pairwise_kernels(Y, Y, metric='laplacian'))-2/(len(X)*len(Y))*np.sum(pairwise_kernels(X, Y, metric='laplacian'))))
'''

def CE_loss(x_hat,x):
    loss=F.binary_cross_entropy(x_hat,x)
    loss=loss.mean()
    return loss