import numpy as np
import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import time
from dolfinx.fem import FunctionSpace
import pyvista
from mpi4py import MPI
from dolfinx import mesh
import tetgen
import meshio
from tqdm import trange
import sys

def volume_2_x(mesh):
    shape=mesh.shape
    mesh=mesh.reshape(-1,mesh.shape[-3],mesh.shape[-2],mesh.shape[-1])
    tmp=np.sum(np.sum(mesh[:,:,:,0],axis=2)*(np.linalg.det(mesh[:,:,1:,1:]-np.expand_dims(mesh[:,:,0,1:],2))/6),axis=1)
    return tmp.reshape(shape[:-3])


def calculate_simulation(name,nodes,elem,bary,write=True):
    start=time.time()
    nodes=nodes-np.min(nodes,axis=0)
    gdim = 3
    shape = "tetrahedron"
    degree = 1
    cell = ufl.Cell(shape, geometric_dimension=gdim)
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))
    domain = mesh.create_mesh(MPI.COMM_WORLD, elem, nodes, domain)
    V = FunctionSpace(domain, ("CG", 2))
    uD = fem.Function(V)
    uD.interpolate(lambda x: np.exp(-((x[0]-bary[0])**2 + (x[1]-bary[1])**2+(x[2]-bary[2])**2)**0.5))
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs)
    #boundary_facets = mesh.locate_entities_boundary(domain, dim=fdim, marker=lambda x:np.isclose(x[2], 0.0))   
    #boundary_dofs = fem.locate_dofs_topological(V=V, entity_dim=fdim, entities=boundary_facets)
    #bc = fem.dirichletbc(value=ScalarType(0), dofs=boundary_dofs, V=V)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V) 
    f = fem.Function(V)
    f.interpolate(lambda x: np.exp(-((x[0]-bary[0])**2 + (x[1]-bary[1])**2+(x[2]-bary[2])**2)))
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    energy=fem.form(u* ufl.dx)
    problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()    
    value=fem.assemble.assemble_scalar(energy)
    u_val=uh.x.array
    if write:
        with io.XDMFFile(domain.comm, "simulations/"+name+".xdmf", "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(uh)
    end=time.time()
    print(end-start)
    return value,u_val

if __name__=="__main__":
    name=sys.argv[1]
    np.random.seed(0)
    NUM_SAMPLES=300
    points=np.load("nn/inference_objects/"+name+".npy").reshape(NUM_SAMPLES,-1,3)
    tets=np.load("data/tetras.npy")
    bary=np.mean(points[0],axis=0)
    value,uh=calculate_simulation("inference_objects/"+name+"_{}".format(0),points[0],tets,bary)
    energy=np.zeros(NUM_SAMPLES)
    u=np.zeros((NUM_SAMPLES,len(uh)))
    energy[0]=value
    u[0]=uh


    for i in trange(1,NUM_SAMPLES):
        value,uh=calculate_simulation("inference_objects/"+name+"_{}".format(i),points[i],tets,bary)
        energy[i]=value
        u[i]=uh

    np.save("simulations/inference_objects/energy_"+name+".npy",energy)
    np.save("simulations/inference_objects/u_"+name+".npy",u)