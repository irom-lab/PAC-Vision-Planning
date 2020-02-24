#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:06:26 2019

@author: sushant
"""

import torch
import torch.nn as nn
from policy import Policy
import warnings
from Parallelizer import Compute_Loss
import json
import time
import sys
warnings.filterwarnings('ignore')


if __name__ == "__main__":

    params = json.load(open('configs/config_multiserver.json'))
    num_trials = params['num_trials']
    multi_server = True
    reg_include = params['reg_include']
    
    import argparse

    def collect_as(coll_type):
        class Collect_as(argparse.Action):
            def __call__(self, parser, namespace, values, options_string=None):
                setattr(namespace, self.dest, coll_type(values))
        return Collect_as

    parser = argparse.ArgumentParser(description='Indiviudal Server Node')
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--num_cpu', type=int, default=1)
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--server_ind', type=int, default=1)
    parser.add_argument('--itr', type=int, default=0)

    args = parser.parse_args()
    
    start_seed = args.start_seed
    num_trials = args.num_seeds
    num_cpu = args.num_cpu
    num_gpu = args.num_gpu
    server_ind = args.server_ind
    itr = args.itr
    
    comp_loss = Compute_Loss(num_trials=num_trials, num_cpu=num_cpu,
                             num_gpu=num_gpu, start_seed=start_seed,
                             multi_server=multi_server)

    policy = Policy()
    num_params = sum(p.numel() for p in policy.parameters())

    # Unpack mu.pt and logvar.pt
    mu_param = nn.ParameterList([nn.Parameter(torch.randn(num_params))])
    logvar_param = nn.ParameterList([nn.Parameter(torch.randn(num_params))])

    mu_param.load_state_dict(torch.load('mu_server.pt'))
    logvar_param.load_state_dict(torch.load('logvar_server.pt'))

    mu = torch.zeros(num_params)
    logvar = torch.zeros(num_params)

    # Copy the parameters of mu and std into the tensors
    mu = mu_param[0]
    logvar = logvar_param[0]

    mu_pr = torch.load('mu_pr.pt', map_location=torch.device("cpu"))
    logvar_pr = torch.load('logvar_pr.pt', map_location=torch.device("cpu"))

    start = time.time()
    emp_cost, grad_mu, grad_logvar, coll_cost, goal_cost = comp_loss.compute(itr,
                                                                            params,
                                                                            mu.clone().detach(),
                                                                            (0.5*logvar).exp().clone().detach(),
                                                                            mu_pr.clone().detach(),
                                                                            logvar_pr.clone().detach(),
                                                                            reg_include)
    
    # Write the output as a dictionary
    output = {}
    output['emp_cost'] = emp_cost
    output['grad_mu'] = grad_mu
    output['grad_logvar'] = grad_logvar
    output['coll_cost'] = coll_cost
    output['goal_cost'] = goal_cost

    # Write the output
    torch.save(output, 'output'+str(server_ind)+'.pt')