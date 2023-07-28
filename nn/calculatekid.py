#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cyberguli
"""
import os
import numpy as np
from pytorch3d import renderer
from pytorch3d import structures
import torch
from tqdm import trange
import matplotlib.pyplot as plt
import meshio
from gaussian_kid import KernelInceptionDistance
from torchvision  import transforms
NUM_SAMPLES=300
point=np.load("data/data.npy")[0]
point=point-np.min(point)
point=point/np.max(point)
pc=structures.pointclouds.Pointclouds(torch.tensor(point.reshape(1,-1,3),dtype=torch.float32))
test=renderer.points.rasterize_points(pc,points_per_pixel=1)[1]

data_arr=torch.zeros(NUM_SAMPLES,256,256,dtype=torch.uint8).cuda()

for i in trange(NUM_SAMPLES):
  point=np.load("data/data.npy")[i]
  pc=structures.pointclouds.Pointclouds(torch.tensor(point.reshape(1,-1,3),dtype=torch.float32).cuda())
  data_arr[i]=transforms.functional.convert_image_dtype(renderer.points.rasterize_points(pc,points_per_pixel=1)[1].reshape(256,256),dtype=torch.uint8)

data_arr=data_arr.reshape(NUM_SAMPLES,1,256,256).repeat(1,3,1,1)
print(data_arr.dtype)
names={"AE",
  "AAE",
  "VAE", 
  "BEGAN",
  "DM",
  "EBM",
  "NF"}

for name in names:
    np.random.seed(0)
    torch.manual_seed(0)
    gen_arr=torch.zeros(NUM_SAMPLES,256,256,dtype=torch.uint8).cuda()
    for i in trange(NUM_SAMPLES):
        point=meshio.read("nn/inference_objects/"+name+"_{}.ply".format(i)).points
        point=point-np.min(point)
        point=point/np.max(point)
        pc=structures.pointclouds.Pointclouds(torch.tensor(point.reshape(1,-1,3),dtype=torch.float32).cuda())
        test=renderer.points.rasterize_points(pc,points_per_pixel=1)
        gen_arr[i]=transforms.functional.convert_image_dtype(renderer.points.rasterize_points(pc,points_per_pixel=1)[1].reshape(256,256),dtype=torch.uint8)
    gen_arr=gen_arr.reshape(NUM_SAMPLES,1,256,256).repeat(1,3,1,1).cuda()
    kid = KernelInceptionDistance(subset_size=NUM_SAMPLES//2,feature=64).cuda()
    kid.update(data_arr, real=True)
    kid.update(gen_arr, real=False)
    kid_mean, kid_std = kid.compute()
    print("mean is :",kid_mean, " and variance is ", kid_std)
    np.save("nn/geometrical_measures/kid_"+name+".npy",kid_mean.detach().cpu().numpy())


