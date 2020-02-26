#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:27:10 2019

@author: sushant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# class Policy(nn.Module):
    
#     def __init__(self, dim_z=20, prim_num=11):
#         super(Policy, self).__init__()
#         # Assume all inputs are square
#         # W is the input size (e.g., 50 for 50X50)
#         # F is the kernel (filter) size (e.g., 10 for 10X10)
#         # P is the zero padding and S is the stride
#         # (W−F+2P)/S+1
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=8,
#                                stride=3, padding=0)
#         # input to conv2: (50-8)/3+1 = 15 (i.e., 32 channels of 15X15)
#         self.conv2 = nn.Conv2d(16, 8, 9, 2)
#         # input to enc1: (15-9)/2+1 = 4 (i.e., 16 channels of 4X4)
#         self.enc = nn.Linear(128, dim_z)
#         # Added 4 for current and goal position
#         self.output = nn.Linear(dim_z+4, prim_num)

#     def forward(self, depth, x, xG):
#         depth = F.elu(self.conv1(depth))
#         depth = F.elu(self.conv2(depth))
#         depth = depth.view(depth.shape[0],-1)
#         depth_feat = F.elu(self.enc(depth))
#         depth_state_aug = torch.cat([depth_feat/1000, x, xG], dim=1)
#         # print(depth_feat.mean().item(), x.mean().item())
#         pd = F.softmax(self.output(depth_state_aug))

#         return pd

# class Policy(nn.Module):
    
#     def __init__(self, dim_z=20, prim_num=101):
#         super(Policy, self).__init__()
#         # Assume all inputs are square
#         # W is the input size (e.g., 50 for 50X50)
#         # F is the kernel (filter) size (e.g., 10 for 10X10)
#         # P is the zero padding and S is the stride
#         # (W−F+2P)/S+1
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=8,
#                                 stride=3, padding=0)
#         # input to conv2: (50-8)/3+1 = 15 (i.e., 32 channels of 15X15)
#         self.conv2 = nn.Conv2d(16, 8, 9, 2)
#         # input to enc1: (15-9)/2+1 = 4 (i.e., 8 channels of 4X4)
#         self.enc1 = nn.Linear(128, dim_z)
#         # self.enc2 = nn.Linear(64, dim_z)
#         # Added 4 for current and goal position
#         self.output1 = nn.Linear(dim_z+2, 50)
#         self.output2 = nn.Linear(50, prim_num)


#     def forward(self, depth, x, xG):
#         depth = F.elu(self.conv1(depth))
#         depth = F.elu(self.conv2(depth))
#         depth = depth.view(depth.shape[0],-1)
#         depth_feat = torch.sigmoid(self.enc1(depth))
#         # depth_feat = torch.sigmoid(self.enc2(depth_feat))
#         depth_state_aug = torch.cat([depth_feat, x], dim=1)
#         depth_state_aug = self.output1(depth_state_aug)
#         pd = F.softmax(self.output2(depth_state_aug))

#         return pd

# class Policy(nn.Module):

#     def __init__(self, dim_z=10, prim_num=51):
#         super(Policy, self).__init__()
#         # Assume all inputs are square
#         # W is the input size (e.g., 50 for 50X50)
#         # F is the kernel (filter) size (e.g., 10 for 10X10)
#         # P is the zero padding and S is the stride
#         # (W−F+2P)/S+1
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=10,
#                                 stride=5, padding=0)
#         # input to conv2: (50-10)/5+1 = 9 (i.e., 32 channels of 15X15)
#         self.conv2 = nn.Conv2d(1, 1, 3, 2)
#         # input to enc1: (9-3)/2+1 = 4 (i.e., 8 channels of 4X4)
#         self.enc1 = nn.Linear(16, dim_z)
#         # self.enc2 = nn.Linear(64, dim_z)
#         # Added 4 for current and goal position
#         self.output = nn.Linear(dim_z+2, prim_num)


#     def forward(self, depth, x, xG):
#         depth = F.elu(self.conv1(depth))
#         depth = F.elu(self.conv2(depth))
#         depth = depth.view(depth.shape[0],-1)
#         depth_feat = torch.sigmoid(self.enc1(depth))
#         # depth_feat = torch.sigmoid(self.enc2(depth_feat))
#         depth_state_aug = torch.cat([depth_feat, x], dim=1)
#         pd = F.softmax(self.output(depth_state_aug))

#         return pd


class Policy_prims(nn.Module):
    
    def __init__(self, dim_z=10, prim_num=50):
        super(Policy, self).__init__()
        # Assume all inputs are square
        # W is the input size (e.g., 50 for 50X50)
        # F is the kernel (filter) size (e.g., 10 for 10X10)
        # P is the zero padding and S is the stride
        # (W−F+2P)/S+1
        self.conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=2, kernel_size=10, stride=5, padding=0),
                                  nn.ELU(),
                                  nn.Conv2d(2, 5, 3, 2),
                                  nn.ELU())
        self.enc = nn.Sequential(nn.Linear(80, 64),
                                  nn.ELU(),
                                  nn.Linear(64, dim_z),
                                  nn.Tanh())
        self.output = nn.Sequential(nn.Linear(dim_z+2, prim_num),
                                    nn.Softmax())


    def forward(self, depth, x, xG):
        depth_feat = self.conv(depth)
        depth_feat = depth_feat.view(depth_feat.shape[0],-1)
        depth_feat = self.enc(depth_feat)
        depth_state_aug = torch.cat([depth_feat, x], dim=1)
        pd = self.output(depth_state_aug)

        return pd


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
                                  # nn.ELU(),
                                  # nn.Conv2d(2, 5, 4, 2),
                                  # (16-4)/2+1 = 7
                                  nn.Tanh())
        # self.enc = nn.Sequential(nn.Linear(100, 64),
        #                           nn.ELU(),
        #                           nn.Linear(64, dim_z),
        #                           nn.ReLU())
        self.output = nn.Sequential(nn.Linear(200+22, out),
                                    nn.Sigmoid())

    def forward(self, depth, angles, ang_vel, base_pos, base_orn):
        depth_feat = self.conv(depth)
        depth_feat = depth_feat.view(depth_feat.shape[0],-1)
        # depth_feat = self.enc(depth_feat)
        inputs = torch.cat([depth_feat, angles, ang_vel, base_pos, base_orn], dim=1)
        action = self.output(inputs)

        return action

class Policy_fc(nn.Module):
    
    def __init__(self, image_size=50, dim_z=10, prim_num=101):
        super(Policy_fc, self).__init__()

        self.enc1 = nn.Linear(3*image_size*image_size, 100)
        self.enc2 = nn.Linear(100, dim_z)
        # Added 4 for current and goal position
        self.output1 = nn.Linear(dim_z+4, 50)
        self.output2 = nn.Linear(50, prim_num)


    def forward(self, depth, x, xG):
        depth = depth.view(-1, depth.shape[0]*depth.shape[1]*depth.shape[2]*depth.shape[3])
        depth_feat = F.relu(self.enc1(depth))
        depth_feat = torch.tanh(self.enc2(depth_feat))
        depth_state_aug = torch.cat([depth_feat, x, xG], dim=1)
        depth_state_aug = torch.tanh(self.output1(depth_state_aug))
        pd = F.softmax(self.output2(depth_state_aug))

        return pd


def load_policy(policy, policy_params):

    count = 0
    for p in policy.parameters():
        num_params_p = p.data.numel()
        p.data = policy_params[count:count+num_params_p].view(p.data.shape)
        count+=num_params_p

    return policy

if __name__ == "__main__":

    policy = Policy()
    num_params = sum(p.numel() for p in policy.parameters())
    policy_params = torch.rand((num_params,))
    print(policy_params.shape)
    # Note: policy is a mutable object, sample(policy,0,0) would work as well
    policy = load_policy(policy, policy_params)


