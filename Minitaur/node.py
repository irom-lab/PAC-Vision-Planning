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
warnings.filterwarnings('ignore')


if __name__ == "__main__":

    params = json.load(open("params.txt"))

    import argparse

    def collect_as(coll_type):
        class Collect_as(argparse.Action):
            def __call__(self, parser, namespace, values, options_string=None):
                setattr(namespace, self.dest, coll_type(values))
        return Collect_as

    def str2bool(v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='Indiviudal Server Node')
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--num_cpu', type=int, default=1)
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--server_ind', type=int, default=1)
    parser.add_argument('--reg_include', type=str2bool, default=True)

    args = parser.parse_args()

    comp_loss = Compute_Loss(num_trials=args.num_seeds, num_cpu=args.num_cpu,
                             num_gpu=args.num_gpu, start_seed=args.start_seed,
                             multi_server=True)

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
    emp_cost, emp_grad_mu, emp_grad_logvar, coll_cost, goal_cost = comp_loss.compute(params,
                                                                                    mu.clone().detach(),
                                                                                    (0.5*logvar).exp().clone().detach(),
                                                                                    mu_pr.clone().detach(),
                                                                                    logvar_pr.clone().detach(),
                                                                                    args.reg_include)
    output = {}
    output['emp_cost'] = emp_cost
    output['emp_grad_mu'] = emp_grad_mu
    output['emp_grad_logvar'] = emp_grad_logvar
    output['coll_cost'] = coll_cost
    output['goal_cost'] = goal_cost

    # Write the output
    torch.save(output, 'output'+str(args.server_ind)+'.pt')