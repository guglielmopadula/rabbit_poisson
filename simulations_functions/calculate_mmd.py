from sklearn.decomposition import PCA
import torch
import gpytorch
import numpy as np
NUMBER_SAMPLES=300
def get_kernel():
    s=gpytorch.kernels.RBFKernel()
    l=[1,2,5,10,20]
    for i in l:
        tmp=gpytorch.kernels.RBFKernel()
        tmp.lengthscale=i
        s=s+tmp
    return s


def relmmd(X,Y):
    s=get_kernel()
    X=X.reshape(X.shape[0],-1)
    Y=Y.reshape(Y.shape[0],-1)
    return (np.mean(s(torch.tensor(X),torch.tensor(X)).to_dense().detach().numpy())+np.mean(s(torch.tensor(Y),torch.tensor(Y)).to_dense().detach().numpy())-2*np.mean(s(torch.tensor(X),torch.tensor(Y)).to_dense().detach().numpy()))/(np.mean(s(torch.tensor(Y),torch.tensor(Y)).to_dense().detach().numpy()))


names=["DM",
       "VAE",
       "AAE",
       "AE",
       "EBM",
       "NF",
       "BEGAN"]
for name in names:
    pca=PCA(n_components=50)
    u_data=np.load("simulations/data/u_data.npy").reshape(NUMBER_SAMPLES,-1)
    u_model=np.load("simulations/inference_objects/u_"+name+".npy").reshape(NUMBER_SAMPLES,-1)
    l=[]
    for i in range(300):
        if np.sum(np.isnan(u_model[i]))<1:
            l.append(i)

    pca.fit(u_data)
    u_model=u_model[l,:]
    print(len(u_model))
    mmd_data=relmmd(pca.transform(u_data.astype(np.float32)).reshape(NUMBER_SAMPLES,-1),pca.transform(u_model.astype(np.float32).reshape(len(u_model),-1)))
    np.save("simulations/inference_objects/mmd_u_"+name+".npy",mmd_data)
    print("MMD of u between data and "+name+" is "+str(mmd_data) )
