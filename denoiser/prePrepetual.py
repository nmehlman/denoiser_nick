import torch
import numpy as np
import argparse
import os
import timeit

from .models import JNDnet

def get_distance(ref, pre):
    pretrained_path = "/home/nmehlman/denoiser_nick/denoiser"
    mname = "/dataset_combined_linear" 
    state = torch.load(pretrained_path+mname+'.pth',map_location="cpu")['state']

    nconv = 14
    nchan = 16
    dist_dp = 0.
    dist_act = 'tshrink'
    ndim0 = 8
    ndim1 = 4
    classif_dp = 0.2
    classif_BN = 2
    classif_act = 'sig'
    model = JNDnet(nconv=nconv,nchan=nchan,dist_dp=dist_dp,dist_act=dist_act,ndim=[ndim0,ndim1],classif_dp=classif_dp,classif_BN=classif_BN,classif_act=classif_act,dev="cuda")
    model.load_state_dict(state)
    model.cuda()
    model.eval()

    #print('forward distance net')
    dist = model.model_dist.forward(ref,pre)
    return dist
