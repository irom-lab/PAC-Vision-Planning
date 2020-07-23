#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:23:19 2019

@author: sushant
"""

import warnings
import time
import torch
import numpy as np
import json
from Parallelizer_compute_C import Compute_Cost_Matrix
warnings.filterwarnings('ignore')

def compute_policy_costs(args):
    # Make a copy of the args for ease of passing
    params = args
    
    # Initialize
    example = args['example']
    num_trials = args['num_trials']
    num_cpu = args['num_cpu']
    num_gpu = args['num_gpu']
    start_seed = args['start_seed']
    save_file_v=args['save_file_v']
    load_prior_from = args['load_prior_from']
    
    # import policy based on the example
    if example == 'quadrotor':
        from policy.quad_policy import Policy
    elif example == 'minitaur':
        from policy.minitaur_policy import Policy
    
    # Generate policy
    policy = Policy()
    num_params = sum(p.numel() for p in policy.parameters())
    print('Number of Neural Network Parameters:', num_params)
    
    # Load prior
    mu_pr = torch.load('Weights/mu_'+str(load_prior_from)+'_best.pt')['0']
    logvar_pr = torch.load('Weights/logvar_'+str(load_prior_from)+'_best.pt')['0']
    
    mu = mu_pr
    logvar = logvar_pr
    
    para = Compute_Cost_Matrix(num_trials, num_cpu, num_gpu, start_seed=start_seed)
    
    start = time.time()
    
    # Compute costs for various runs
    emp_cost_stack = para.compute(0, params, mu.clone().detach(), (0.5*logvar).exp().clone().detach())
    
    print("Time to compute all costs:", time.time()-start)
    
    C = emp_cost_stack.numpy()
    np.save("Weights/C_"+save_file_v+".npy",C)
    

if __name__ == "__main__":

    import argparse

    def collect_as(coll_type):
        class Collect_as(argparse.Action):
            def __call__(self, parser, namespace, values, options_string=None):
                setattr(namespace, self.dest, coll_type(values))
        return Collect_as

    parser = argparse.ArgumentParser(description='Compute Cost Matrix')
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--num_envs', type=int, default=100)
    parser.add_argument('--num_policies', type=int, default=50)

    args_con = parser.parse_args()
    args = json.load(open(args_con.config_file))

    args['start_seed'] = args_con.start_seed
    args['num_trials'] = args_con.num_envs
    args['num_policy_eval'] = args_con.num_policies
    
    compute_policy_costs(args)