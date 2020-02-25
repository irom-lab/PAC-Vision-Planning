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
from policy import Policy as Policy
from head_node import run_servers
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
        self.num_trials = args['num_trials']
        self.num_cpu = args['num_cpu']
        self.num_gpu = args['num_gpu']
        self.reg_include = args['reg_include']
        self.lr_mu = args['lr_mu']
        self.lr_logvar = args['lr_logvar']
        self.reg_grad_wt = args['reg_grad_wt']
        self.start_seed = args['start_seed']
        self.save_file_v=args['save_file_v']
        self.server_list = args['server_list']
        self.delta = args['delta']
        self.load_weights_from = args['load_weights_from']
        self.load_weights = args['load_weights']
        self.load_optimizer = args['load_optimizer']
        self.load_prior = args['load_prior']
        self.load_prior_from = args['load_prior_from']

        # Generate policy
        self.policy = Policy()
        self.num_params = sum(p.numel() for p in self.policy.parameters())
        print('Number of Neural Network Parameters:', self.num_params)

        # Establish prior
        if self.load_prior:
            self.mu_pr = torch.load('Weights/mu_'+str(self.load_prior_from)+'_best.pt')['0']
            self.logvar_pr = torch.load('Weights/logvar_'+str(self.load_prior_from)+'_best.pt')['0']
            # self.logvar_pr = torch.log(self.logvar_pr.exp()*4)
        else:            
            self.mu_pr = torch.zeros(self.num_params)
            self.logvar_pr = torch.log(torch.ones(self.num_params)*100)

        # Load necessary params to all servers
        if len(self.server_list) == 0:
            self.multi_server = False
        else:
            self.multi_server = True

        if self.multi_server:
            torch.save(self.mu_pr, 'mu_pr.pt')
            torch.save(self.logvar_pr, 'logvar_pr.pt')
            load_list = ['mu_pr.pt', 'logvar_pr.pt', sys.argv[1]]
            for i in range(len(load_list)):
                for j in range(len(self.server_list)):
                    os.system('./put_on_server.sh '+self.server_list[j]+' '+load_list[i])
                    
        self.mu = self.mu_pr
        self.logvar = self.logvar_pr
        
    def opt(self):

        mu = self.mu
        logvar = self.logvar

        para = Compute_Loss(self.num_trials, self.num_cpu, self.num_gpu, start_seed=self.start_seed)
        
        start = time.time()

        if self.multi_server:
            torch.save(self.mu.state_dict(), 'mu_server.pt')
            torch.save(self.logvar.state_dict(), 'logvar_server.pt')
            # Compute costs for various runs
            emp_cost, emp_grad_mu, emp_grad_logvar, coll_cost, goal_cost = run_servers(self.server_list,
                                                                                       self.num_trials,
                                                                                       self.num_cpu,
                                                                                       self.num_gpu,
                                                                                       self.reg_include)
        else:
            # Compute costs for various runs
            emp_cost, coll_cost, goal_cost, emp_cost_stack = para.compute(0,
                                                                          self.params,
                                                                          mu.clone().detach(),
                                                                          (0.5*logvar).exp().clone().detach(),
                                                                          self.mu_pr.clone().detach(),
                                                                          self.logvar_pr.clone().detach(),
                                                                          self.reg_include)
            
        print("Time:", time.time()-start)
            
        C = emp_cost_stack.numpy()
        np.save("Weights/C_"+self.save_file_v+".npy",C)
        cost_policywise = emp_cost_stack.mean(dim=0)
        print("Smallest cost:", cost_policywise.min().item())
                
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
