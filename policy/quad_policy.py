#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:27:10 2019

@author: sushant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    '''Big Net'''

    def __init__(self, dim_z=10, prim_num=25):
        super(Policy, self).__init__()
        # Assume all inputs are square
        # W is the input size (e.g., 50 for 50X50)
        # F is the kernel (filter) size (e.g., 10 for 10X10)
        # P is the zero padding and S is the stride
        # (W−F+2P)/S+1
        self.conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=2, kernel_size=4, stride=2, padding=0),
                                  # (50-4)/2+1 = 24
                                  nn.ELU(),
                                  nn.Conv2d(in_channels=2, out_channels=1, kernel_size=2, stride=1),
                                  # (24-2)/1+1 = 23
                                  nn.ELU())
        self.output = nn.Sequential(nn.Linear(23**2, prim_num),
                                    nn.ELU(),
                                    nn.Linear(prim_num, prim_num),
                                    nn.Tanh(),
                                    )

    def forward(self, depth):
        depth_feat = self.conv(depth)
        depth_feat = depth_feat.view(depth_feat.shape[0],-1)
        pd = 1e3 * self.output(depth_feat)

        return pd
    
class Filter(nn.Module):
    '''Filter'''

    def __init__(self):
        super(Filter, self).__init__()
        # Assume all inputs are square
        # W is the input size (e.g., 50 for 50X50)
        # F is the kernel (filter) size (e.g., 10 for 10X10)
        # P is the zero padding and S is the stride
        # (W−F+2P)/S+1
        self.conv = nn.Sequential(nn.Conv2d(in_channels=1, 
                                            out_channels=1, 
                                            kernel_size=5, 
                                            stride=5, 
                                            padding=0,
                                            bias=False))
        self.kernel_size = 5
        
    def forward(self, depth):
        depth_filtered = self.conv(depth[:,:,9:34,13:38])/(self.kernel_size**2)
        return depth_filtered

def load_policy(policy, policy_params):

    count = 0
    for p in policy.parameters():
        num_params_p = p.data.numel()
        p.data = policy_params[count:count+num_params_p].view(p.data.shape)
        count+=num_params_p

    return policy