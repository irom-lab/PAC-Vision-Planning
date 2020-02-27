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

    def __init__(self, dim_z=10, out=4):
        super(Policy, self).__init__()
        # Assume all inputs are square
        # W is the input size (e.g., 50 for 50X50)
        # F is the kernel (filter) size (e.g., 10 for 10X10)
        # P is the zero padding and S is the stride
        # (W−F+2P)/S+1
        self.conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5, stride=5, padding=0),
                                  # (50-5)/5+1 = 10
                                  nn.Tanh())

        self.output = nn.Sequential(nn.Linear(200+22, out),
                                    nn.Sigmoid())

    def forward(self, depth, angles, ang_vel, base_pos, base_orn):
        depth_feat = self.conv(depth)
        depth_feat = depth_feat.view(depth_feat.shape[0],-1)
        inputs = torch.cat([depth_feat, angles, ang_vel, base_pos, base_orn], dim=1)
        action = self.output(inputs)

        return action

def load_policy(policy, policy_params):

    count = 0
    for p in policy.parameters():
        num_params_p = p.data.numel()
        p.data = policy_params[count:count+num_params_p].view(p.data.shape)
        count+=num_params_p

    return policy