import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx.fem import FunctionSpace
import pyvista
from mpi4py import MPI
from dolfinx import mesh
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
import tetgen
import meshio
import basix
from tqdm import trange
from scipy.interpolate import RBFInterpolator
from time import time
from dolfinx.fem.petsc import LinearProblem

def volume_2_x(mesh):
    shape=mesh.shape
    mesh=mesh.reshape(-1,mesh.shape[-3],mesh.shape[-2],mesh.shape[-1])
    tmp=np.sum(np.sum(mesh[:,:,:,0],axis=2)*(np.linalg.det(mesh[:,:,1:,1:]-np.expand_dims(mesh[:,:,0,1:],2))/6),axis=1)
    return tmp.reshape(shape[:-3])


def calculate_simulation(name,nodes,elem,bary,write=True):
    nodes=nodes-np.min(nodes,axis=0)
    gdim = 3
    degree = 1
    tensor_element = basix.ufl.element("Lagrange", basix.CellType(basix._basixcpp.CellType.tetrahedron), degree, shape=(gdim,))
    domain = mesh.create_mesh(MPI.COMM_WORLD, elem, nodes, tensor_element)
    V = fem.functionspace(domain, ("Lagrange",1))
    uD = fem.Function(V)
    uD.interpolate(lambda x: np.exp(-((x[0]-bary[0])**2 + (x[1]-bary[1])**2+(x[2]-bary[2])**2)**0.5))
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V) 
    f = fem.Function(V)
    f.interpolate(lambda x: np.exp(-((x[0]-bary[0])**2 + (x[1]-bary[1])**2+(x[2]-bary[2])**2)))
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    energy=fem.form(u* ufl.dx)
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()    
    value=fem.assemble.assemble_scalar(energy)
    u_val=uh.x.array
    if write:
        with io.XDMFFile(domain.comm, "simulations/"+name+".xdmf", "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(uh)
    return value,u_val

if __name__=="__main__":
    np.random.seed(0)
    NUM_SAMPLES=300
    points=np.load("data/data.npy").reshape(NUM_SAMPLES,-1,3)
    tets=np.load("data/tetras.npy")
    bary=np.mean(points[0],axis=0)
    value,uh=calculate_simulation("data/bunny_{}".format(0),points[0],tets,bary)
    energy_data=np.zeros(NUM_SAMPLES)
    u_data=np.zeros((NUM_SAMPLES,len(uh)))
    energy_data[0]=value
    u_data[0]=uh


    for i in trange(1,NUM_SAMPLES):
        value,uh=calculate_simulation("data/bunny_{}".format(i),points[i],tets,bary)
        energy_data[i]=value
        u_data[i]=uh

    np.save("simulations/data/energy_data.npy",energy_data)
    np.save("simulations/data/u_data.npy",u_data)