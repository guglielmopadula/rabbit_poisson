import meshio
import numpy as np
from cpffd import *
from tqdm import trange
NUM_SAMPLES=600
np.random.seed(0)
p=np.load("data/points.npy")
np.random.seed(0)
triangles=np.load("data/tetras.npy")
def scale_normalize(points):
    minim=np.min(points,axis=0)
    points=points-minim
    scale=np.max(points)
    points=points/scale
    return points,minim,scale

def restore(points,scale,minim):
    return points*scale+minim

p,minim,scale=scale_normalize(p)
n_x=3
n_y=3
n_z=3
mask=np.ones((n_x,n_y,n_z),dtype=int)
mask[:,:,0]=0
print(np.sum(mask))
M=np.eye(np.sum(mask)*3,dtype=np.float32)
latent=np.zeros((NUM_SAMPLES,3,int(np.sum(mask))))

indices_c=np.arange(n_x*n_y*n_z)[mask.reshape(-1).astype(bool)]
indices_c.sort()
vpffd=cpffd.BPFFD((n_x,n_y,n_z))

a=0.2
for i in trange(NUM_SAMPLES):
    vpffd.array_mu_x=a*np.random.rand(*vpffd.array_mu_x.shape)*np.arange(n_z).reshape(1,1,-1)
    vpffd.array_mu_y=a*np.random.rand(*vpffd.array_mu_y.shape)*np.arange(n_z).reshape(1,1,-1)
    vpffd.array_mu_z=a*np.random.rand(*vpffd.array_mu_z.shape)*np.arange(n_z).reshape(1,1,-1)
    latent[i,0]=vpffd.array_mu_x[:,:,1:].reshape(-1)
    latent[i,1]=vpffd.array_mu_y[:,:,1:].reshape(-1)
    latent[i,2]=vpffd.array_mu_z[:,:,1:].reshape(-1)
    pdef=vpffd.barycenter_ffd_adv(p,M,indices_c)
    pdef=restore(pdef,scale,minim)
    meshio.write_points_cells("data/bunny_{}.ply".format(i),pdef,[])


np.save("data/dffd_latent.npy",latent)
