#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:40:10 2019

@author: sushant
"""
import torch

def compute_grad_ES(costs, epsilons, std, method='ES'):
    # Compute gradient of 1-loss

    num_policy_eval = int(costs.numel()/2) # costs for eps and -eps

    if method=='utility':
        '''Scale by utility function u instead of the loss'''
        fit_index = costs.sort().indices
        epsilons = epsilons[fit_index]

        # Fitness rank transformation
        u = torch.zeros(num_policy_eval*2)
        for i in range(num_policy_eval*2):
            u[i] = torch.max(torch.Tensor([torch.Tensor([num_policy_eval+1]).log() - torch.Tensor([i+1]).log(), 0]))
        u /= u.sum()
        u -= 1./(2*num_policy_eval)

        loss = u
        grad_mu = (torch.matmul(torch.transpose(epsilons,0,1), loss) / std)/epsilons.shape[0]
        grad_std = (torch.matmul(torch.transpose(epsilons**2-torch.ones_like(epsilons),0,1), loss) / std)/epsilons.shape[0]
        logvar = torch.log(std.pow(2))
        grad_logvar = 0.5 * grad_std * (logvar*0.5).exp()
        # The direction with the lowest cost is given the highest utility,
        # so effectively we are finding the negative of the gradient
        return -grad_mu, -grad_logvar

    if method=='ES':
        '''Pick num_fit best epsilon and scale by 1-loss function'''
        num_fit_frac = 1
        fit_index = costs.sort().indices
        num_fit = int(2*num_policy_eval*num_fit_frac)
        fit_index = fit_index[:num_fit]

        loss = costs[fit_index]
        epsilons = epsilons[fit_index]
        grad_mu = (torch.matmul(torch.transpose(epsilons,0,1), loss) / std)/epsilons.shape[0]
        grad_std = (torch.matmul(torch.transpose(epsilons**2-torch.ones_like(epsilons),0,1), loss) / std)/epsilons.shape[0]
        logvar = torch.log(std.pow(2))
        grad_logvar = 0.5 * grad_std * (logvar*0.5).exp()
        return grad_mu, grad_logvar