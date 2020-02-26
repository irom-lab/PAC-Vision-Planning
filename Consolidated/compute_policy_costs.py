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
import scipy
import json
import os
import sys
from optimize_PAC_bound import optimize_PAC_bound, optimize_quad_PAC_bound_bisection, kl_inverse
from Parallelizer_compute_C import Compute_Loss
warnings.filterwarnings('ignore')

class compute_policy_costs:

    def __init__(self, args):

        # Make a copy of the args for ease of passing
        self.params = args

        # Initialize
        self.example = args['example']
        self.num_trials = args['num_trials']
        self.num_cpu = args['num_cpu']
        self.num_gpu = args['num_gpu']
        self.start_seed = args['start_seed']
        self.save_file_v=args['save_file_v']
        self.delta = args['delta']
        self.load_prior_from = args['load_prior_from']
        
        # import policy based on the example
        if self.example == 'quadrotor':
            from policy.quad_policy import Policy
        elif self.example == 'minitaur':
            from policy.minitaur_policy import Policy

        # Generate policy
        self.policy = Policy()
        self.num_params = sum(p.numel() for p in self.policy.parameters())
        print('Number of Neural Network Parameters:', self.num_params)

        # Load prior
        self.mu_pr = torch.load('Weights/mu_'+str(self.load_prior_from)+'_best.pt')['0']
        self.logvar_pr = torch.load('Weights/logvar_'+str(self.load_prior_from)+'_best.pt')['0']

        self.mu = self.mu_pr
        self.logvar = self.logvar_pr
        
    def pac_bayes_opt(self):

        mu = self.mu
        logvar = self.logvar

        para = Compute_Loss(self.num_trials, self.num_cpu, self.num_gpu, start_seed=self.start_seed)
        
        start = time.time()

        # Compute costs for various runs
        emp_cost_stack = para.Compute_Cost_Matrix(0, self.params, mu.clone().detach(), (0.5*logvar).exp().clone().detach())

        print("Time:", time.time()-start)

        C = emp_cost_stack.numpy()
        print(C)
        np.save("Weights/C_"+self.save_file_v+".npy",C)
        # cost_policywise = emp_cost_stack.mean(dim=0)
        # print("Smallest cost:", cost_policywise.min().item())
                
        num_policy_eval = self.params['num_policy_eval']
        p0 = np.ones(num_policy_eval)/num_policy_eval

        print('========================')
        print('    Mc Allester Opt     ')
        print('========================')
        tau_opt, p_opt, taus, new_emp_cost = optimize_PAC_bound(C, p0, self.delta)
        print("PAC Bound:", tau_opt)
        print("New Emp Cost:", new_emp_cost)
        r = (np.sum(scipy.special.kl_div(p_opt, p0)) + np.log(2*np.sqrt(self.num_trials)/self.delta))/(2*self.num_trials)
        print("R:",r)
        pac_bound = kl_inverse(new_emp_cost, 2*r)
        np.save("Weights/p_"+self.save_file_v+".npy", p_opt)
        
        print("KL-inv PAC bound:", pac_bound)
        
        
        print('========================')
        print('        Quad Opt        ')
        print('========================')
        tau_opt, p_opt, new_emp_cost = optimize_quad_PAC_bound_bisection(C, p0, self.delta)
        print("Quad PAC Bound:", tau_opt)
        print("New Emp Cost:", new_emp_cost)
        r = (np.sum(scipy.special.kl_div(p_opt, p0)) + np.log(2*np.sqrt(self.num_trials)/self.delta))/(2*self.num_trials)
        print("R:",r)
        pac_bound = new_emp_cost + r**0.5
        print("McAllester:", pac_bound)
        quad_pac_bound = kl_inverse(new_emp_cost, 2*r)
        
        if quad_pac_bound < pac_bound:
            np.save("Weights/p_"+self.save_file_v+".npy", p_opt)
        
        print("KL-inv PAC bound:", quad_pac_bound)

if __name__ == "__main__":

    args={}
    args = json.load(open(sys.argv[1]))

    train1 = compute_policy_costs(args)
    train1.opt()
