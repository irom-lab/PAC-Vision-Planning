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
from policy import Policy as Policy
from head_node import run_servers
import json
import os
import sys
from ES_grad import compute_grad_ES
from Parallelizer import Compute_Loss
from visualize import cost_spread_train, weight_spread
warnings.filterwarnings('ignore')

class train:

    def __init__(self, args):

        # Make a copy of the args for ease of passing
        self.params = args

        # Initialize
        self.num_trials = args['num_trials']
        self.num_itr = args['num_itr']
        self.num_cpu = args['num_cpu']
        self.num_gpu = args['num_gpu']
        self.reg_include = args['reg_include']
        self.lr_mu = args['lr_mu']
        self.lr_logvar = args['lr_logvar']
        self.reg_grad_wt = args['reg_grad_wt']
        self.itr_start = args['itr_start']
        self.start_seed = args['start_seed']
        self.save_file_v=args['save_file_v']
        self.server_list = args['server_list']
        self.delta = args['delta']
        self.load_weights_from = args['load_weights_from']
        self.load_weights = args['load_weights']
        self.load_optimizer = args['load_optimizer']
        self.load_prior = args['load_prior']
        self.load_prior_from = args['load_prior_from']
        self.eps_shared = args['eps_shared']

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

        # Load necessary params to all servers
        if len(self.server_list) == 0:
            self.multi_server = False
        else:
            self.multi_server = True

        if self.multi_server:
            torch.save(self.mu_pr, 'mu_pr.pt')
            torch.save(self.logvar_pr, 'logvar_pr.pt')
            load_list = ['mu_pr.pt', 'logvar_pr.pt']
            folder_path = args['folder_path']
            for i in range(len(load_list)):
                for j in range(len(self.server_list)):
                    os.system('./put_on_server.sh '+self.server_list[j]+' '+load_list[i]+' '+folder_path)
            for i in range(len(self.server_list)):
                os.system('./put_on_server.sh '+self.server_list[i]+' '+sys.argv[1]+' '+folder_path+'configs/')
                    
        # Initialize the posterior distribution
        self.mu = nn.ParameterList([nn.Parameter(torch.randn(self.num_params))])
        self.logvar = nn.ParameterList([nn.Parameter(torch.randn(self.num_params))])

        if self.load_weights is True:
            # Load posterior distribution from file
            self.mu.load_state_dict(torch.load('Weights/mu_'+str(self.load_weights_from)+'.pt'))
            self.logvar.load_state_dict(torch.load('Weights/logvar_'+str(self.load_weights_from)+'.pt'))
            # self.logvar = nn.ParameterList([nn.Parameter(self.logvar_pr.clone())])

        else:
            self.mu = nn.ParameterList([nn.Parameter(self.mu_pr.clone())])
            self.logvar = nn.ParameterList([nn.Parameter(self.logvar_pr.clone())])

        # Initialize the gradients, by default they are set to None
        self.mu.grad = torch.randn_like(self.mu[0])
        self.logvar.grad = torch.randn_like(self.logvar[0])

    def kld_grad(self, mu, logvar):
        '''McAllester regularizer
        '''
        mu.requires_grad_(True)
        logvar.requires_grad_(True)
        # Compute kld with the prior
        kld = -0.5 * torch.sum(1 + logvar-self.logvar_pr - (mu-self.mu_pr).pow(2)/self.logvar_pr.exp() - (logvar-self.logvar_pr).exp())
        kld.backward()
        return kld.detach(), mu.grad.detach(), logvar.grad.detach()

    def reg_grad(self, mu, logvar):
        '''McAllester regularizer
        '''
        mu.requires_grad_(True)
        logvar.requires_grad_(True)
        # Compute kld with the prior
        kld = -0.5 * torch.sum(1 + logvar-self.logvar_pr - (mu-self.mu_pr).pow(2)/self.logvar_pr.exp() - (logvar-self.logvar_pr).exp())
        reg = ((kld + torch.log(torch.Tensor([2*(self.num_trials**0.5)/self.delta]))) / (2*self.num_trials)).pow(0.5)
        reg.backward()
        return reg.detach(), mu.grad.detach(), logvar.grad.detach()

    def quad_bound_grad(self, emp, reg, emp_grad, reg_grad):
        term = (emp + reg).pow(0.5)
        grad = 2 * (term + reg.pow(0.5)) * ( (emp_grad + reg_grad)/(2*term) + reg_grad/(2*reg.pow(0.5)) )
        return grad

    def kld_quad_bound(self, emp_cost, mu, logvar):
        mu.requires_grad_(True)
        logvar.requires_grad_(True)
        # Compute kld with the prior
        kld = -0.5 * torch.sum(1 + logvar-self.logvar_pr - (mu-self.mu_pr).pow(2)/self.logvar_pr.exp() - (logvar-self.logvar_pr).exp())
        reg = (kld + torch.log(torch.Tensor([2*(self.num_trials**0.5)/self.delta]))) / (2*self.num_trials)
        reg.backward()
        return reg.detach(), mu.grad.detach(), logvar.grad.detach()

    def R_grad(self, mu, logvar, num_eval=500):
        std = (0.5*logvar).exp()
        # kld = -0.5 * torch.sum(1 + logvar-self.logvar_pr - (mu-self.mu_pr).pow(2)/self.logvar_pr.exp() - (logvar-self.logvar_pr).exp())

        epsilon = torch.randn((num_eval, mu.numel()))
        epsilon = torch.cat([epsilon, -epsilon], dim=0)
        theta = mu + std*epsilon

        reg, _, _ = self.reg_grad(mu.clone().detach(), logvar.clone().detach())

        reg_loss = torch.sum(0.5*(self.logvar_pr-logvar) + (theta-self.mu_pr).pow(2)/(2*(self.logvar_pr.exp())) - (theta-mu).pow(2)/(2*(logvar.exp())), dim=1) \
                         / (4*self.num_trials*reg)
        
        reg = reg_loss.mean()
        reg_grad_mu, reg_grad_logvar = compute_grad_ES(reg_loss, epsilon, std, method=self.params['grad_method'])
        
        return reg, reg_grad_mu, reg_grad_logvar
        
    def opt(self):

        optimizer = optim.Adam([ {'params': self.mu, 'lr': self.lr_mu},
                                 {'params': self.logvar, 'lr': self.lr_logvar} ])
        if self.load_optimizer:
            optimizer.load_state_dict(torch.load('optim_state/optimizer_'+self.save_file_v+'_current.pt'))

        cost_min = self.params['cost_min']
        writer = SummaryWriter(log_dir='runs/summary_'+self.save_file_v, flush_secs=10)
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
            
            if self.multi_server:
                torch.save(self.mu.state_dict(), 'mu_server.pt')
                torch.save(self.logvar.state_dict(), 'logvar_server.pt')
                # Compute costs for various runs
                emp_cost, emp_grad_mu, emp_grad_logvar, coll_cost, goal_cost = run_servers(i, self.params)
            else:
                # Compute costs for various runs
                emp_cost, emp_grad_mu, emp_grad_logvar, coll_cost, goal_cost = para.compute(i,
                                                                                            self.params,
                                                                                            mu.clone().detach(),
                                                                                            (0.5*logvar).exp().clone().detach(),
                                                                                            self.mu_pr.clone().detach(),
                                                                                            self.logvar_pr.clone().detach(),
                                                                                            self.reg_include,
                                                                                            self.eps_shared)

            # Only log the latest image, otherwise tensorboard's memory consumption will rapidly grow
            fig_mu, fig_std = weight_spread(mu.clone().detach(), (0.5*logvar).exp().clone().detach())
            writer.add_figure('Mean Spread', fig_mu, 0)
            writer.add_figure('Std Spread', fig_std, 0)

            # fig = cost_spread_train(emp_cost, 1)
            # writer.add_figure('Cost Spread', fig, 0)

            emp_cost = emp_cost.sum()/self.num_trials
            coll_cost = coll_cost.sum()/self.num_trials
            goal_cost = goal_cost.sum()/self.num_trials
            
            cost_compare = emp_cost.item()

            if self.reg_include:
                reg, reg_grad_mu, reg_grad_logvar = self.kld_quad_bound(emp_cost, mu.clone().detach(), logvar.clone().detach())
                PAC_cost = ((emp_cost + reg).pow(0.5) + reg.pow(0.5)).pow(2)
                
                cost_compare = PAC_cost.item()
                
                if PAC_cost < 0.25:
                    self.reg_method = 'Quad'
                else:
                    self.reg_method = 'McAllester'
                    
                if self.reg_method == 'McAllester':
                    # McAllester PAC-Bayes Bound
                    reg, reg_grad_mu, reg_grad_logvar = self.reg_grad(mu.clone().detach(), logvar.clone().detach())
                    PAC_cost = emp_cost + reg
                    grad_mu = emp_grad_mu + self.reg_grad_wt * reg_grad_mu
                    grad_logvar = emp_grad_logvar + self.reg_grad_wt * reg_grad_logvar
                elif self.reg_method == 'Quad':
                    # Quadratic PAC-Bayes Bound
                    grad_mu = self.quad_bound_grad(emp_cost, reg, emp_grad_mu, self.reg_grad_wt * reg_grad_mu)
                    grad_logvar = self.quad_bound_grad(emp_cost, reg, emp_grad_logvar, self.reg_grad_wt * reg_grad_logvar)
            else:
                grad_mu = emp_grad_mu
                grad_logvar = emp_grad_logvar

            '''emp_grad is the pac_grad here. The above if condition only computed
            to print necessary stuff'''
            grad_mu = emp_grad_mu
            grad_logvar = emp_grad_logvar

            grad_mu_norm = torch.norm(grad_mu, p=2).item()
            grad_logvar_norm = torch.norm(grad_logvar, p=2).item()
            
            # Load the gradients into the parameters
            self.mu[0].grad = grad_mu
            self.logvar[0].grad = grad_logvar

            if self.reg_include:
                writer.add_scalars('Loss', {'PAC-Bayes':PAC_cost,
                                            'Train':emp_cost,
                                            'Regularizer': reg.item(),
                                            'Goal Cost': goal_cost.item(),
                                            'Collision Cost': coll_cost.item()}, i)

                print('Itr: {}, time:{:.1f} s, PAC-Bayes Cost: {:.3f}, Train Cost: {:.3f}, '
                      'Reg: {:.3f}, Goal Cost: {:.3f}, Coll Cost: {:.3f}, Mean: {:.3f}, '
                      'Std: {:.3f}'.format(i, time.time()-start, PAC_cost.item(),
                      emp_cost.item(), reg.item(), goal_cost.item(), coll_cost.item(),
                      mu.mean().item(), (0.5*logvar).exp().mean().item() ))


            else:
                writer.add_scalars('Loss', {'Train':emp_cost,
                                            'Goal Cost': goal_cost.item(),
                                            'Collision Cost': coll_cost.item()}, i)

                print('Itr: {}, time:{:.1f} s, Train Cost: {:.3f}, Goal Cost: {:.3f}, '
                      'Coll Cost: {:.3f}, Mean: {:.3f}, Std: {:.3f}'.format(i, time.time()-start,
                      emp_cost.item(), goal_cost.item(), coll_cost.item(),
                      mu.mean().item(), (0.5*logvar).exp().mean().item() ))


            writer.add_scalars('Gradients', {'mu grad': grad_mu_norm,
                                             'logvar grad': grad_logvar_norm}, i)

            # Save the mean and log of variance
            if cost_min > cost_compare:
                torch.save(self.mu.state_dict(), 'Weights/mu_'+self.save_file_v+'_best.pt')
                torch.save(self.logvar.state_dict(), 'Weights/logvar_'+self.save_file_v+'_best.pt')
                # writer.add_figure('Best Cost Spread', fig, 0)
                cost_min = cost_compare

            torch.save(self.mu.state_dict(), 'Weights/mu_'+self.save_file_v+'_current.pt')
            torch.save(self.logvar.state_dict(), 'Weights/logvar_'+self.save_file_v+'_current.pt')
            torch.save(optimizer.state_dict(), 'optim_state/optimizer_'+self.save_file_v+'_current.pt')

            # Update the parameters
            optimizer.step()
            
            mu_step_size = (self.mu[0]-mu).norm()
            logvar_step_size = (self.logvar[0]-logvar).norm()

            writer.add_scalars('Optimization Step Size', {'Mean Step':mu_step_size,
                                                          'Logvar Step':logvar_step_size}, i)
            
            state_dict = json.load(open(sys.argv[1]))
            state_dict['itr_start'] = i+1
            state_dict['load_weights'] = True
            state_dict['load_optimizer'] = True
            state_dict['load_weights_from'] = self.save_file_v+'_current'
            state_dict['cost_min'] = cost_min
            json.dump(state_dict, open(sys.argv[1], 'w'), indent=4)

            if self.reg_include:
                del grad_mu_norm, grad_logvar_norm, emp_cost, PAC_cost, goal_cost, coll_cost, mu, logvar
                del reg, reg_grad_logvar, reg_grad_mu, emp_grad_logvar, emp_grad_mu
            else:
                del grad_mu_norm, grad_logvar_norm, emp_cost, goal_cost, coll_cost, mu, logvar
                del emp_grad_logvar, emp_grad_mu


        writer.close()

if __name__ == "__main__":

    args={}
    args = json.load(open(sys.argv[1]))

    train1 = train(args)
    train1.opt()
