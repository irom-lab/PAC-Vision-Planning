#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 16:07:09 2020

@author: sushant
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import warnings
warnings.filterwarnings('ignore')

class quadrotor_prior_dataloader(Dataset):

    def __init__(self, folder_path):
        self.depth = np.load(folder_path+'prior_images.npy')
        self.prim = np.load(folder_path+'prior_labels.npy')
        self.len = self.depth.shape[0]

    def __getitem__(self, index):
        return self.depth[index,:,:], self.prim[index,:]
    
    def __len__(self):
        return self.len
    
def compute_grad(torch_seed, policy, mu, std, num_policy_eval, device="cpu"):
    
    mu = mu.to(device)
    std = std.to(device)
    policy = policy.to(device)
    grad_mu = torch.zeros_like(mu, device=device)
    grad_std = torch.zeros_like(mu, device=device)
    prims = np.load('prior_data/prior_labels.npy')
    batch_size = prims.shape[0]
    exp_loss = 0
    prim_count = torch.zeros(25, device=device)
    train_in_batch = torch.utils.data.DataLoader(quadrotor_prior_dataloader('prior_data/'),
                                                 batch_size=batch_size, shuffle=True, num_workers=1)
    
    def loss_func(output, labels):
        # true = torch.ones(output.shape[0],25)*(1/25)
        
        # true = torch.zeros(output.shape[0],25)
        # indices = torch.randint(low=0, high=25, size=[1,true.shape[0]])
        # for i in range(indices.numel()):
        #     true[i,indices[0,i]] = 1
        
        # return F.mse_loss(output, true, reduction='sum')
        return F.binary_cross_entropy(output, labels, reduction='sum')

    # Set the seed to 0 to fix the epsilons and true and check if the optimization
    # at least converges on that
    # torch.manual_seed(0)
    torch.manual_seed(torch_seed)
    epsilon = torch.randn((num_policy_eval, mu.numel()), device=device)
    for i in range(num_policy_eval):
        theta = mu + std*epsilon[i,:]
        loss_theta = 0
        grad_theta = torch.zeros_like(mu)
        
        count = 0
        for p in policy.parameters():
            num_params_p = p.data.numel()
            p.data = theta[count:count+num_params_p].view(p.data.shape)
            count+=num_params_p
            
        for batch_idx, (depth, labels) in enumerate(train_in_batch):
            policy.zero_grad()
            depth = depth.unsqueeze(1).float().to(device)
            labels = labels.float().to(device)
            output = policy(depth)
            loss = loss_func(output, labels)
            loss.backward()
            loss_theta += loss.detach()
            prims = output.max(dim=1).indices
            for j in range(prims.numel()):
                prim_count[prims[j]] += 1
                
            count = 0
            grad_theta_temp = torch.zeros_like(mu)
            for p in policy.parameters():
                num_params_p = p.data.numel()
                grad_theta_temp[count:count+num_params_p] = p.grad.view(-1).detach()
                count+=num_params_p
            grad_theta += grad_theta_temp
        # scale the gradient by the number of depth maps in the dataset because
        # the gradient is computed for the loss as a sum, not mean
        grad_mu += grad_theta/len(train_in_batch.dataset)
        grad_std += (grad_theta/len(train_in_batch.dataset))*epsilon[i,:]
        exp_loss += loss_theta/len(train_in_batch.dataset)
        
    grad_mu /= num_policy_eval
    grad_std /= num_policy_eval
    exp_loss /= num_policy_eval
    prim_count /= num_policy_eval * len(train_in_batch.dataset)
    
    grad_logvar = 0.5 * grad_std * std
    
    return -grad_mu.to("cpu"), -grad_logvar.to("cpu"), exp_loss.to("cpu"), prim_count.to("cpu")