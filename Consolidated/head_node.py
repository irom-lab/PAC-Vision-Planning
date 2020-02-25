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
import os
warnings.filterwarnings('ignore')

def parallel_server(server_ind, server_ip, folder_path, seed_start, num_seeds, num_cpu, num_gpu, itr):
    
    os.system('./run_server.sh '+str(server_ip)+' '+str(seed_start)+' '+str(num_seeds)
              +' '+str(num_cpu)+' '+str(num_gpu)+' '+str(server_ind)+' '+str(itr)+' '+str(folder_path))

def run_servers(itr, params):
    
    server_ip_list = params['server_list']
    num_envs = params['num_trials']
    num_cpu = params['num_cpu']
    num_gpu = params['num_gpu']
    folder_path = params['folder_path']
    
    process = []
    num_servers = len(server_ip_list)
    batch = [0] * num_servers
    for i in range(num_envs):
        batch[i % num_servers] += 1
    pos = 0
    
    for j in range(num_servers):
        process.append(mp.Process(target=parallel_server, args=(j, server_ip_list[j], 
                                                                folder_path, pos, 
                                                                batch[j], num_cpu, 
                                                                num_gpu, itr)))
        pos += batch[j]
        process[j].start()
    
    for j in range(num_servers):
        process[j].join()
    
    
    policy = Policy()
    num_params = sum(p.numel() for p in policy.parameters())
    
    # Collect the epsilons along with cost (how to keep them separate from other environments?)
    grad_mu = torch.zeros(num_params)
    grad_logvar = torch.zeros(num_params)

    emp_cost = []
    coll_cost = []
    goal_cost = []
    for i in range(num_servers):
        output = torch.load('output'+str(i)+'.pt')
        grad_mu += output['grad_mu']
        grad_logvar += output['grad_logvar']
        emp_cost.extend(output['emp_cost'].view(1,output['emp_cost'].numel()))
        coll_cost.extend(output['coll_cost'].view(1,output['coll_cost'].numel()))
        goal_cost.extend(output['goal_cost'].view(1,output['goal_cost'].numel()))

    emp_cost = torch.cat(emp_cost)
    goal_cost = torch.cat(goal_cost)
    coll_cost = torch.cat(coll_cost)

    grad_mu /= num_envs
    grad_logvar /= num_envs

    return emp_cost, grad_mu, grad_logvar, coll_cost, goal_cost