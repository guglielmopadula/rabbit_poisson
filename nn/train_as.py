from ezyrb import Database,RBF, GPR, KNeighborsRegressor, RadiusNeighborsRegressor, Linear, ANN, ReducedOrderModel, POD, AE, PODAE
from  sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, ConstantKernel
from sklearn.gaussian_process.kernels import RBF as RBFKernel    
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np
from tqdm import trange
np.random.seed(0)
from athena.active import ActiveSubspaces

NUM_SAMPLES=300
NUM_TRAIN_SAMPLES=250
parameters=np.load("data/dffd_latent.npy").reshape(3000,-1)[:300,]
snapshot_1=np.load("simulations/data/u_data.npy").reshape(NUM_SAMPLES,-1)
np.save("simulations/inference_objects/u_AS.npy",snapshot_1)
snapshot_2=np.load("simulations/data/energy_data.npy").reshape(NUM_SAMPLES,-1)
np.save("simulations/inference_objects/energy_AS.npy",snapshot_2)

train_index=np.random.choice(NUM_SAMPLES, NUM_TRAIN_SAMPLES, replace=False)
test_index=np.setdiff1d(np.arange(NUM_SAMPLES),train_index)
pca=PCA(n_components=100)
pca.fit(snapshot_1)
snapshot_1=pca.transform(snapshot_1)


asub_1 = ActiveSubspaces(dim=5,method='local')

asub_1.fit(parameters,snapshot_1)
parameters_1=asub_1.transform(parameters)[0]

np.save("nn/inference_objects/AS_latent_u.npy",parameters_1)

asub_2 = ActiveSubspaces(dim=5,method='local')

asub_2.fit(parameters,snapshot_2)
parameters_2=asub_2.transform(parameters)[0]

np.save("nn/inference_objects/AS_latent_energy.npy",parameters_2)



