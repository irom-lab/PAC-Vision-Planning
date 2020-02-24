#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:27:49 2019

@author: sushant
"""

import torch
import torch.optim as optim
import time

def compute_grad(std, loss, epsilons):
    grad_mu = (torch.matmul(torch.transpose(epsilons,0,1), loss) / std)/epsilons.shape[0]
    grad_std = (torch.matmul(torch.transpose(epsilons**2-torch.ones_like(epsilons),0,1), loss) / std)/epsilons.shape[0]
    logvar = torch.log(std.pow(2))
    grad_logvar = 0.5 * grad_std * (logvar*0.5).exp()
    return grad_mu, grad_logvar

def RBO(std, pert_costs, epsilons, unpert_cost, num_trials):
    y = pert_costs - unpert_cost
    Z = epsilons*std
    v = torch.randn_like(std, requires_grad=True)
    optimizer = optim.Adam([v], 1e-1)
    for i in range(100):
        optimizer.zero_grad()
        loss = ((y-torch.matmul(Z,v))/(2*num_trials)).norm()**2 + 0.1 * v.norm()**2
        loss.backward()
        optimizer.step()

    grad_mu = v.detach()
    grad_logvar = torch.zeros_like(grad_mu)
    return grad_mu, grad_logvar


def loss1(x):
    """Loss: 
        (x-1)^2 for x>0
        (x+1)^2 for x<0
    """
    a = torch.sign(x)+torch.ones_like(x)
    b = torch.sign(x)-torch.ones_like(x)
    return 0.5*a*(x-1)**2 - 0.5*b*(x+1)**2

def loss2(x):
    """Loss:
        1 for x<-1
        0 for -1<x<1
        1 for x>1
    """
    return 0.5*(torch.sign(torch.abs(x)-1) + 1)

def opt(num_itr, num_trials, lr):
    mu = (torch.randn(1)-10).to("cuda")
    logvar = torch.zeros(1).to("cuda")
    epsilons = torch.randn((num_trials, 1)).to("cuda")
    for i in range(num_itr):
        start = time.time()
        std = (0.5*logvar).exp()
        x = mu + std*epsilons
        costs = loss1(x)
        unpert_cost = loss1(mu)
        grad_mu, grad_logvar = RBO(std, costs, epsilons, unpert_cost, num_trials)
        # grad_mu, grad_logvar = compute_grad(std, costs, epsilons)
        mu += -lr*grad_mu.item()
        logvar += -lr*grad_logvar.item()
        print('Iteration: {}, time:{:.1f} s, Average cost: {:.3f}, Mean: {:.3f}, SD: {:.3f}'.format(
          i, time.time()-start, costs.sum().item()/num_trials, mu.to("cpu").item(), (0.5*logvar).exp().to("cpu").item()))
    return mu, (0.5*logvar).exp()

if __name__ == "__main__":
    mu, std = opt(num_itr=1000, num_trials=100, lr=1e-1)