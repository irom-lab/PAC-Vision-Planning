#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 09:26:03 2019

@author: sushant
"""

import torch
from policy import Policy, Filter
import json
import warnings
from Simulator import Simulator
from Environment import TestEnvironment
import numpy as np
import sys
import time
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
        safecircle = params['safecircle']
        alpha = params['alpha']
        
        num_policies = posterior.shape[0]

        policy_eval_costs = torch.zeros(num_policy_eval)
        
        # creating objects
        sim = Simulator(comp_len=comp_len, prim_horizon=prim_horizon, alpha=alpha,
                        dt=time_step, t_horizon=t_horizon, device=device)  # new env for this thread to use
        env = TestEnvironment(r_lim, num_obs, safecircle=safecircle, parallel=False, gui=True,
                          y_max=y_max, y_min=y_min, x_min=x_min, x_max=x_max)
        batch_costs = torch.zeros(num_policy_eval)
        epsilon = torch.randn((num_policies, mu.numel()))

        # Unlike parallelizer we do not need to reset the robot because we only
        # run one environment per env object
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
                                                                        record_vid=True,
                                                                        vid_num=seed)

            # cost, collision_cost, goal_cost = sim.deterministic_prior(env, DepthFilter, image_size=image_size, plot_line=True)

            
            if collision_cost == 0:
                collision_flag = 1
                collision_counter += 1

            policy_eval_costs[j] = torch.Tensor([cost])

            # print('Evaluation: {} | Cost: {:.3f}, Goal Cost: {:.3f}, Coll Cost: {:.3f}'
            #       .format(j, cost, goal_cost, collision_cost))
            batch_costs[j] = torch.Tensor([cost])
        env.p.disconnect()  # clean up instance
        # print("Average Cost:", batch_costs.mean().item())
        return batch_costs, collision_flag, collision_counter

if __name__ == "__main__":

    params = {}
    params = json.load(open(sys.argv[1]))
    params['num_policy_eval'] = 1
    save_file_v = params['save_file_v']
    load_prior_from = params['load_prior_from']
    if len(sys.argv) > 2:
        env_num_start = int(sys.argv[2])
        num_envs = int(sys.argv[3])
    else:
        num_envs = params['num_trials']
        env_num_start = params['start_seed']
    
    device = torch.device("cuda")

    policy = Policy()
    policy = policy.to(device)
    DepthFilter = Filter()
    for p in DepthFilter.parameters():
        p.data = torch.ones_like(p.data)
    DepthFilter = DepthFilter.to(device)
    num_params = sum(p.numel() for p in policy.parameters())
    print(num_params)
    mu = torch.zeros(num_params)
    logvar = torch.zeros(num_params)

    mu_dict = torch.load('Weights/mu_'+str(load_prior_from)+'_best.pt')
    logvar_dict = torch.load('Weights/logvar_'+str(load_prior_from)+'_best.pt')

    mu = mu_dict['0']
    logvar = logvar_dict['0']
    # logvar = torch.ones(num_params) * torch.log(torch.ones(1)*100)

    posterior = np.load("Weights/p_"+save_file_v+".npy")    
    posterior /= posterior.sum() # The result from cvx does not sum to 1 exactly, it is 1 + epsilon, where epsilon -> 0

    std = (0.5*logvar).exp()

    collision_free_counter = 0
    total_cost = 0
    start = time.time()
    for i in range(env_num_start,env_num_start+num_envs):
        print("==============")
        print("Environment "+str(i+1)+":")
        print("==============")
        cost, collision_flag, collision_counter = test(params, policy, DepthFilter, device, mu, std, posterior, i)
        total_cost += cost.mean()
        collision_free_counter += collision_flag
        print("Env Num:", i+1)
        print("Emp Cost So Far:", total_cost.item()/(i-env_num_start+1))
        # print("Collision Free Percentage:", collision_free_counter*100/(i+1))
        # print("Num of Collision Free Rollouts:", collision_counter)
    print(collision_free_counter)
    print('Emp_cost:', total_cost.item()/num_envs)
    print("Total Time taken:", time.time()-start)
    
    np.save('Weights/Emp_cost_'+save_file_v+'.npy', np.array(total_cost.item()/num_envs))