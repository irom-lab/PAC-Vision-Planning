#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:23:19 2019

@author: sushant
"""

import multiprocessing as mp
import warnings
import torch
from policy import Policy
warnings.filterwarnings('ignore')

def parallel_server(server_ind, server_ip, seed_start, num_seeds, num_cpu, num_gpu, reg_include):

    import os
    os.system('./run_server.sh '+str(server_ip)+' '+str(seed_start)+' '+str(num_seeds)
              +' '+str(num_cpu)+' '+str(num_gpu)+' '+str(server_ind)+' '+str(reg_include))

def run_servers(server_ip_list, num_envs, num_cpu, num_gpu, reg_include):
    process = []
    num_servers = len(server_ip_list)
    batch = [0] * num_servers
    for i in range(num_envs):
        batch[i % num_servers] += 1
    pos = 0
    
    for j in range(num_servers):
        process.append(mp.Process(target=parallel_server, args=(j, server_ip_list[j], pos, batch[j], num_cpu, num_gpu, reg_include)))
        pos += batch[j]
        process[j].start()
    
    for j in range(num_servers):
        process[j].join()
    
    
    policy = Policy()
    num_params = sum(p.numel() for p in policy.parameters())
    
    # Collect the epsilons along with cost (how to keep them separate from other environments?)
    grad_mu = torch.zeros(num_params)
    grad_logvar = torch.zeros(num_params)
    emp_cost = torch.zeros(1)
    coll_cost = torch.zeros(1)
    goal_cost = torch.zeros(1)
    for i in range(num_servers):
        output = torch.load('output'+str(i)+'.pt')
        grad_mu += output['emp_grad_mu']
        grad_logvar += output['emp_grad_logvar']
        emp_cost += output['emp_cost']
        coll_cost += output['coll_cost']
        goal_cost += output['goal_cost']
    emp_cost /= num_envs
    coll_cost /= num_envs
    goal_cost /= num_envs
    grad_mu /= num_envs
    grad_logvar /= num_envs

    return emp_cost, grad_mu, grad_logvar, coll_cost, goal_cost


if __name__ == '__main__':
    server_ip_list = ['xx.xx.xxxx', 'xx.xx.xxxx']
    num_envs = 10
    run_servers(server_ip_list, num_envs)



