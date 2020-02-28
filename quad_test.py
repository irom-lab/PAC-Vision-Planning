#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 09:26:03 2019

@author: sushant
"""

import torch
from policy.quad_policy import Policy, Filter
import json
import warnings
from envs.Quad_Simulator import Simulator
from envs.Quad_Env import Environment
import numpy as np
import sys
warnings.filterwarnings('ignore')

def test(params, policy, DepthFilter, device, mu, std, posterior, seed):
        time_step = params['time_step']
        image_size = params['image_size']
        t_horizon = params['t_horizon']
        r_lim = params['r_lim']
        num_obs = params['num_obs']
        y_max=params['y_max']
        y_min=params['y_min']
        x_min=params['x_min']
        x_max=params['x_max']
        comp_len = params['comp_len']
        prim_horizon = params['prim_horizon']
        num_policy_eval = params['num_policy_eval']
        alpha = params['alpha']
        
        num_policies = posterior.shape[0]

        policy_eval_costs = torch.zeros(num_policy_eval)
        
        # creating objects
        env = Environment(r_lim, num_obs, parallel=True,
                          gui=True, y_max=y_max, y_min=y_min, x_min=x_min, x_max=x_max)
        sim = Simulator(comp_len=comp_len, prim_horizon=prim_horizon, alpha=alpha,
                        dt=time_step, t_horizon=t_horizon, device=device)
        batch_costs = torch.zeros(num_policy_eval)
        epsilon = torch.randn((num_policies, mu.numel()))

        # Unlike parallelizer we do not need to reset the robot because we only
        # run one env per env-object
        np.random.seed(seed)
        env.generate_safe_initial_env(1.25)
        
        collision_flag = 0
        collision_counter = 0

        np.random.seed(int(2 ** 32 * np.random.random_sample()))
        policy_eval = np.random.choice(num_policies, num_policy_eval, p=posterior) 
        
        for j in range(num_policy_eval):
            
            policy_params = mu + std*epsilon[policy_eval[j],:]
            policy_params = policy_params.to(device)

            # LOAD POLICY_PARAMS
            count = 0
            for p in policy.parameters():
                num_params_p = p.data.numel()
                p.data = policy_params[count:count+num_params_p].view(p.data.shape)
                count+=num_params_p

            cost, collision_cost, goal_cost = sim.simulate_controller(env, policy, DepthFilter, gen_new_env=False, rem_old_env=False,
                                                                      RealTimeSim=True, image_size=image_size, plot_line=True,
                                                                        record_vid=False, vid_num=seed)

            if collision_cost == 0:
                collision_flag = 1
                collision_counter += 1

            policy_eval_costs[j] = torch.Tensor([cost])

            batch_costs[j] = torch.Tensor([cost])
        env.p.disconnect()  # clean up instance
        return batch_costs, collision_flag, collision_counter

if __name__ == "__main__":

    import argparse

    def collect_as(coll_type):
        class Collect_as(argparse.Action):
            def __call__(self, parser, namespace, values, options_string=None):
                setattr(namespace, self.dest, coll_type(values))
        return Collect_as

    parser = argparse.ArgumentParser(description='PAC-Bayes Optimization')
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--start_seed', type=int, default=10000)
    parser.add_argument('--num_envs', type=int, default=1)
    args = parser.parse_args()

    env_num_start = args.start_seed
    num_envs = args.num_envs

    params = json.load(open(args.config_file))
    params['num_policy_eval'] = 1
    save_file_v = params['save_file_v']
    load_prior_from = params['load_prior_from']
    
    device = torch.device("cuda")

    policy = Policy()
    policy = policy.to(device)
    DepthFilter = Filter()
    for p in DepthFilter.parameters():
        p.data = torch.ones_like(p.data)
    DepthFilter = DepthFilter.to(device)
    num_params = sum(p.numel() for p in policy.parameters())
    print("Number of parameters in the policy:", num_params)
    mu = torch.zeros(num_params)
    logvar = torch.zeros(num_params)

    mu_dict = torch.load('Weights/mu_'+str(load_prior_from)+'_best.pt')
    logvar_dict = torch.load('Weights/logvar_'+str(load_prior_from)+'_best.pt')

    mu = mu_dict['0']
    logvar = logvar_dict['0']

    posterior = np.load("Weights/p_"+save_file_v+".npy")    
    posterior /= posterior.sum() # The result from cvx does not sum to 1 exactly, it is 1 + epsilon, where epsilon -> 0

    std = (0.5*logvar).exp()

    total_cost = 0
    for i in range(env_num_start,env_num_start+num_envs):
        print("==============")
        print("Environment "+str(i+1)+":")
        print("==============")
        cost, collision_flag, collision_counter = test(params, policy, DepthFilter, device, mu, std, posterior, i)
        total_cost += cost.mean()
        print("Env Num:", i+1)
        print("Emp Cost So Far:", total_cost.item()/(i-env_num_start+1))
    print('Emp_cost:', total_cost.item()/num_envs)