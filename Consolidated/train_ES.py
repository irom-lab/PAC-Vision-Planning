#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:23:19 2019

@author: sushant
"""

import warnings
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import json
import sys
from Parallelizer import Compute_Loss
from visualize import weight_spread
warnings.filterwarnings('ignore')

class train:

    def __init__(self, args):

        # Make a copy of the args for ease of passing
        self.params = args

        # Initialize
        self.example = args['example']
        self.num_trials = args['num_trials']
        self.num_itr = args['num_itr']
        self.num_cpu = args['num_cpu']
        self.num_gpu = args['num_gpu']
        self.lr_mu = args['lr_mu']
        self.lr_logvar = args['lr_logvar']
        self.itr_start = args['itr_start']
        self.start_seed = args['start_seed']
        self.save_file_v=args['save_file_v']
        self.delta = args['delta']
        self.load_weights = args['load_weights']
        self.load_weights_from = args['load_weights_from']
        self.load_optimizer = args['load_optimizer']
        self.load_prior = args['load_prior']
        self.load_prior_from = args['load_prior_from']
        self.logging = args['logging']
        
        # import policy based on the example
        if self.example == 'quadrotor':
            from policy.quad_policy import Policy
        elif self.example == 'minitaur':
            from policy.minitaur_policy import Policy

        # Generate policy
        self.policy = Policy()
        self.num_params = sum(p.numel() for p in self.policy.parameters())
        print('Number of Neural Network Parameters:', self.num_params)

        # Establish prior
        if self.load_prior:
            self.mu_pr = torch.load('Weights/mu_'+str(self.load_prior_from)+'_best.pt')['0']
            self.logvar_pr = torch.load('Weights/logvar_'+str(self.load_prior_from)+'_best.pt')['0']
            self.logvar_pr = torch.log(self.logvar_pr.exp()*4)
        else:            
            self.mu_pr = torch.zeros(self.num_params)
            self.logvar_pr = torch.log(torch.ones(self.num_params)*4)

        # Initialize the posterior distribution
        self.mu = nn.ParameterList([nn.Parameter(torch.randn(self.num_params))])
        self.logvar = nn.ParameterList([nn.Parameter(torch.randn(self.num_params))])

        if self.load_weights is True:
            # Load posterior distribution from file
            self.mu.load_state_dict(torch.load('Weights/mu_'+str(self.load_weights_from)+'.pt'))
            self.logvar.load_state_dict(torch.load('Weights/logvar_'+str(self.load_weights_from)+'.pt'))
        else:
            # Use the prior as the posterior
            self.mu = nn.ParameterList([nn.Parameter(self.mu_pr.clone())])
            self.logvar = nn.ParameterList([nn.Parameter(self.logvar_pr.clone())])

        # Initialize the gradients, by default they are set to None
        self.mu.grad = torch.randn_like(self.mu[0])
        self.logvar.grad = torch.randn_like(self.logvar[0])

        # Initialize the tensorboard writer for logging
        if self.logging:
            self.writer = SummaryWriter(log_dir='runs/summary_'+self.save_file_v, flush_secs=10)

        
    def logger(self, itr, mu, logvar, grad_mu_norm, grad_logvar_norm, 
               mu_step_size, logvar_step_size, emp_cost, cost_min):
        ''' Logs: 
            (1) the training data to tensorboard for each iteration
            (2) updates to the config file to allow restarting the training if 
                terminated midway
        '''
        # For plots: only log the latest image, otherwise tensorboard's memory consumption grows rapidly
        fig_mu, fig_std = weight_spread(mu.clone().detach(), (0.5*logvar).exp().clone().detach())
        self.writer.add_figure('Mean Spread', fig_mu, 0)
        self.writer.add_figure('Std Spread', fig_std, 0)
        self.writer.add_scalars('Loss', {'Train':emp_cost}, itr)
        self.writer.add_scalars('Gradient Norm', {'mu grad': grad_mu_norm,
                                                  'logvar grad': grad_logvar_norm}, itr)
        self.writer.add_scalars('Optimization Step Size', {'Mean Step':mu_step_size,
                                                           'Logvar Step':logvar_step_size}, itr)
        
        # Log updates to the config file
        state_dict = json.load(open(sys.argv[1]))
        state_dict['itr_start'] = itr+1
        state_dict['load_weights'] = True
        state_dict['load_optimizer'] = True
        state_dict['load_weights_from'] = self.save_file_v+'_current'
        state_dict['cost_min'] = cost_min
        json.dump(state_dict, open(sys.argv[1], 'w'), indent=4)

    def opt(self):

        optimizer = optim.Adam([ {'params': self.mu, 'lr': self.lr_mu},
                                 {'params': self.logvar, 'lr': self.lr_logvar} ])

        # Load optimizer state
        if self.load_optimizer:
            optimizer.load_state_dict(torch.load('optim_state/optimizer_'+self.save_file_v+'_current.pt'))

        # Load cost_min from the config file
        cost_min = self.params['cost_min']
        
        # Instantiate the Parallelizer for computing cost
        para = Compute_Loss(self.num_trials, self.num_cpu, self.num_gpu, start_seed=self.start_seed)

        for i in range(self.itr_start, self.num_itr):

            optimizer.zero_grad()
            start = time.time()

            # Initialization of tensor "copies" of self.mu and self.std
            mu = torch.zeros(self.num_params)
            logvar = torch.zeros(self.num_params)

            # Copy the parameters self.mu and self.std into the tensors
            mu = self.mu[0].clone()
            logvar = self.logvar[0].clone()
            
            # Compute costs for various runs
            emp_cost, grad_mu, grad_logvar = para.compute(i,
                                                          self.params,
                                                          mu.clone().detach(),
                                                          (0.5*logvar).exp().clone().detach())

            
            # Compute gradient norms
            grad_mu_norm = torch.norm(grad_mu, p=2).item()
            grad_logvar_norm = torch.norm(grad_logvar, p=2).item()
            
            # Load the gradients into the parameters
            self.mu[0].grad = grad_mu
            self.logvar[0].grad = grad_logvar

            # Save the mean, log of variance, and the current optimizer state
            torch.save(self.mu.state_dict(), 'Weights/mu_'+self.save_file_v+'_current.pt')
            torch.save(self.logvar.state_dict(), 'Weights/logvar_'+self.save_file_v+'_current.pt')
            torch.save(optimizer.state_dict(), 'optim_state/optimizer_'+self.save_file_v+'_current.pt')

            # Save the mean, log of variance for the "best" iteration
            if cost_min > emp_cost.item():
                torch.save(self.mu.state_dict(), 'Weights/mu_'+self.save_file_v+'_best.pt')
                torch.save(self.logvar.state_dict(), 'Weights/logvar_'+self.save_file_v+'_best.pt')
                cost_min = emp_cost.item()

            # Update the parameters
            optimizer.step()
            
            # Monitor step size, helpful in debugging
            mu_step_size = (self.mu[0]-mu).norm()
            logvar_step_size = (self.logvar[0]-logvar).norm()

            # Print iterations
            print('Itr: {}, time:{:.1f} s, Emp Cost: {:.3f}'.format(i, 
                  time.time()-start, emp_cost.item()))

            # Log to tensorboard and the config file
            if self.logging:
                self.logger(i, mu, logvar, grad_mu_norm, grad_logvar_norm, 
                             mu_step_size, logvar_step_size, emp_cost, cost_min)
            
            del grad_mu_norm, grad_logvar_norm, emp_cost, mu, logvar

        if self.logging:
            self.writer.close()

if __name__ == "__main__":

    args={}
    args = json.load(open(sys.argv[1]))

    train1 = train(args)
    train1.opt()