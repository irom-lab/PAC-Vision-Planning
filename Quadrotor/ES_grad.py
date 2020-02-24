#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:40:10 2019

@author: sushant
"""
import torch
import torch.optim as optim
import time

def compute_grad_ES(costs, epsilons, std, num_fit_frac=1., method='utility', unpert_cost=torch.zeros(1)):
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
        # grad_mu /= loss.std()
        # grad_std /= loss.std()
        logvar = torch.log(std.pow(2))
        grad_logvar = 0.5 * grad_std * (logvar*0.5).exp()
        # The direction with the lowest cost is given the highest utility,
        # so effectively we are finding the negative of the gradient
        return -grad_mu, -grad_logvar

    if method=='ES':
        '''Pick num_fit best epsilon and scale by 1-loss function'''
        fit_index = costs.sort().indices
        num_fit = int(2*num_policy_eval*num_fit_frac)
        fit_index = fit_index[:num_fit]

        if num_fit_frac < 1:
            loss = 1. - costs[fit_index]
        else:
            loss = costs[fit_index]
        epsilons = epsilons[fit_index]
        grad_mu = (torch.matmul(torch.transpose(epsilons,0,1), loss) / std)/epsilons.shape[0]
        grad_std = (torch.matmul(torch.transpose(epsilons**2-torch.ones_like(epsilons),0,1), loss) / std)/epsilons.shape[0]
        # grad_mu /= loss.std()
        # grad_std /= loss.std()
        logvar = torch.log(std.pow(2))
        grad_logvar = 0.5 * grad_std * (logvar*0.5).exp()
        if num_fit_frac < 1:
            grad_mu *= -1
            grad_logvar *= -1
        return grad_mu, grad_logvar

    if method=='NES':
        '''Pick num_fit best epsilon and scale by 1-loss function'''
        fit_index = costs.sort().indices
        num_fit = int(2*num_policy_eval*num_fit_frac)
        fit_index = fit_index[:num_fit]

        if num_fit_frac < 1:
            loss = 1. - costs[fit_index]
        else:
            loss = costs[fit_index]
        epsilons = epsilons[fit_index]
        grad_mu_log_post = (epsilons/std.pow(2)).to('cuda')
        grad_std_log_post = ((epsilons.pow(2) - std.pow(2))/std.pow(3)).to('cuda')

        # start = time.time()
        F_mu = torch.zeros([grad_mu_log_post.shape[1], grad_mu_log_post.shape[1]], requires_grad=False, device='cuda')
        F_std = torch.zeros([grad_std_log_post.shape[1], grad_std_log_post.shape[1]], requires_grad=False, device='cuda')
        for i in range(int(grad_mu_log_post.shape[0])):
            F_mu += torch.matmul(grad_mu_log_post[i,:].unsqueeze(1), grad_mu_log_post[i,:].unsqueeze(0))
            F_std += torch.matmul(grad_std_log_post[i,:].unsqueeze(1), grad_std_log_post[i,:].unsqueeze(0))
        F_mu /= grad_mu_log_post.shape[0]
        F_std /= grad_std_log_post.shape[0]
        
        alpha = 1e-3
        
        F_mu_inv = torch.inverse(F_mu.to('cpu') + alpha*torch.eye(F_mu.shape[0]))
        F_std_inv = torch.inverse(F_std.to('cpu') + alpha*torch.eye(F_std.shape[0]))

        # Verifying that the Fischer information matrix is positive definite
        # _, s, _ = torch.svd(F_mu_inv)
        # print(s.min())

        F_mu_inv = F_mu_inv.to('cpu')
        F_std_inv = F_std_inv.to('cpu')

        # batch multiplication; tries to allocate a lot of memory ~50-70 GB
        # F_mu = torch.bmm(grad_mu_log_post.unsqueeze(2), grad_mu_log_post.unsqueeze(1)).mean(0)
        # F_logvar = torch.bmm(grad_std_log_post.unsqueeze(2), grad_std_log_post.unsqueeze(1)).mean(0)

        # grad_mu = (torch.matmul(torch.transpose(epsilons,0,1), loss) / std)/epsilons.shape[0]
        # grad_std = (torch.matmul(torch.transpose(epsilons**2-torch.ones_like(epsilons),0,1), loss) / std)/epsilons.shape[0]
        grad_mu = torch.matmul(F_mu_inv, (torch.matmul(torch.transpose(epsilons,0,1), loss) / std)/epsilons.shape[0])
        grad_std = torch.matmul(F_std_inv, (torch.matmul(torch.transpose(epsilons**2-torch.ones_like(epsilons),0,1), loss) / std)/epsilons.shape[0])
        # print(torch.dot(grad_mu, grad_mu_orig)/(grad_mu.norm()*grad_mu_orig.norm()))
        logvar = torch.log(std.pow(2))
        grad_logvar = 0.5 * grad_std * (logvar*0.5).exp()
        if num_fit_frac < 1:
            grad_mu *= -1
            grad_logvar *= -1
        return grad_mu, grad_logvar


    if method=='RBO':
        '''See equation (5) in https://arxiv.org/abs/1903.02993'''
        y = costs - unpert_cost
        Z = epsilons*std
        v = torch.randn_like(std, requires_grad=True)
        optimizer = optim.Adam([v], 1e-1)
        for i in range(300):
            optimizer.zero_grad()
            loss = ((y-torch.matmul(Z,v))/(4*num_policy_eval)).norm()**2 + 0.1 * v.norm()**2
            loss.backward()
            optimizer.step()

        grad_mu = v.detach()
        grad_logvar = torch.zeros_like(grad_mu)
        return grad_mu, grad_logvar