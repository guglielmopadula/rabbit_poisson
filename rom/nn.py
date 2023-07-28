from ezyrb import Database,RBF, GPR, KNeighborsRegressor, RadiusNeighborsRegressor, Linear, ANN, ReducedOrderModel, POD, AE, PODAE
from  sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, ConstantKernel
from sklearn.gaussian_process.kernels import RBF as RBFKernel    
import torch.nn as nn
from sklearn.decomposition import PCA
import numpy as np
from tqdm import trange
np.random.seed(0)

names=["VAE",
       "AAE",
       "AE",
       "BEGAN",
       "DM",
       "EBM",
       "NF",
       "AS"
       ]

NUM_SAMPLES=300
NUM_TRAIN_SAMPLES=250
for name in names:
    np.random.seed(0)
    parameters=np.load("nn/inference_objects/"+name+"_latent.npy").reshape(NUM_SAMPLES,-1)
    snapshot_1=np.load("simulations/inference_objects/u_"+name+".npy").reshape(NUM_SAMPLES,-1)
    snapshot_2=np.load("simulations/inference_objects/energy_"+name+".npy").reshape(NUM_SAMPLES,-1)

    l=[]
    for i in range(300):
        if np.sum(np.isnan(snapshot_1[i]))<1:
            l.append(i)

    parameters=parameters[l,:]
    snapshot_1=snapshot_1[l,:]
    snapshot_2=snapshot_2[l,:]
    NUM_SAMPLES=len(snapshot_1)

    train_index=np.random.choice(NUM_SAMPLES, NUM_TRAIN_SAMPLES, replace=False)
    test_index=np.setdiff1d(np.arange(NUM_SAMPLES),train_index)
    parameters_train=parameters[train_index]
    parameters_test=parameters[test_index]
    snapshot_1_train=snapshot_1[train_index]
    snapshot_2_train=snapshot_2[train_index]
    snapshot_1_test=snapshot_1[test_index]
    snapshot_2_test=snapshot_2[test_index]



    db1=Database(parameters,snapshot_1)
    db2=Database(parameters,snapshot_2)

    db_t={"u": db1, "energy": db2}

    train={"u":[parameters_train,snapshot_1_train],"energy":[parameters_train,snapshot_2_train] }
    test={"u":[parameters_test,snapshot_1_test],"energy":[parameters_test,snapshot_2_test] }


    podae=PODAE(POD('svd'),AE([200, 100, 10], [10, 100, 200], nn.Tanh(), nn.Tanh(), 5000))

    approximations = {
        'RBF': RBF(),
        'GPR': GPR(),
        'KNeighbors': KNeighborsRegressor(),
        'ANN': ANN([2000, 2000], nn.Tanh(), 1000,l2_regularization=0.03,lr=0.001),
    }


    train_error=np.zeros((2,4))
    test_error=np.zeros((2,4))

    for approxname, approxclass in approximations.items():
        j=list(approximations.keys()).index(approxname)
        approxclass.fit(train["energy"][0],train["energy"][1])
        train_error[1,j]=np.linalg.norm(approxclass.predict(train["energy"][0]).reshape(-1,1)-train["energy"][1])/np.linalg.norm(train["energy"][1])
        test_error[1,j]=np.linalg.norm(approxclass.predict(test["energy"][0]).reshape(-1,1)-test["energy"][1])/np.linalg.norm(test["energy"][1])



    for approxname, approxclass in approximations.items():
        j=list(approximations.keys()).index(approxname)
        rom = ReducedOrderModel(db_t["u"], podae, approxclass)
        rom.fit()
        train_error[0,j]=np.linalg.norm(rom.predict(train["u"][0])-train["u"][1])/np.linalg.norm(train["u"][1])
        test_error[0,j]=np.linalg.norm(rom.predict(test["u"][0])-test["u"][1])/np.linalg.norm(test["u"][1])


    approximations=list(approximations.keys())
    db_t=list(db_t.keys())
    #f = open("./rom/graphs_txt/"+name+"_.txt", "a")
    for i in range(len(db_t)):
        for j in range(len(approximations)):
            print("Training error of "+str(approximations[j])+" over " + str(db_t[i]) +" is "+str(train_error[i,j]))
            print("Test error of "+str(approximations[j])+" over " + str(db_t[i]) +" is "+str(test_error[i,j]))

    np.save("./simulations/inference_objects/"+name+"_rom_err_train.npy",train_error)
    np.save("./simulations/inference_objects/"+name+"_rom_err_test.npy",test_error)