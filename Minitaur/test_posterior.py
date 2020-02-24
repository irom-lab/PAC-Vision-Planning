#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 09:26:03 2019

@author: sushant
"""
import torch
from policy import Policy as Policy
import json
import warnings
from Environment import Environment
import numpy as np
import os
import sys
import time
warnings.filterwarnings('ignore')

def test(params, policy, device, mu, std, posterior, seed):
        image_size = params['image_size']
        max_angle = params['max_angle']
        comp_len = params['comp_len']
        prim_horizon = params['prim_horizon']
        num_policy_eval = params['num_policy_eval']
        alpha = params['alpha']
        time_step = params['time_step']
        goal = params['goal']

        num_policies = posterior.shape[0]

        batch_costs = torch.zeros(num_policy_eval)
        
        torch.manual_seed(0)
        epsilon = torch.randn((num_policies, mu.numel()))
        # epsilon = torch.cat([epsilon, -epsilon], dim=0)

        env = Environment(max_angle, gui=True)
        
        env.goal = goal
        
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

            if j>0:
                env.p.removeBody(env.terrain)
                env.minitaur_env.reset()
            np.random.seed(seed)
            env.terrain = env.generate_steps()

            # logID = env.p.startStateLogging(loggingType=env.p.STATE_LOGGING_VIDEO_MP4, fileName='video'+str(seed)+'.mp4')
            cost, fall_cost, goal_cost, end_position = env.execute_policy(policy,
                                                            env.goal,
                                                            alpha,
                                                            time_step=time_step,
                                                            comp_len=comp_len,
                                                            prim_horizon=prim_horizon,
                                                            image_size=image_size,
                                                            device=device,
                                                            record_vid=False,
                                                            vid_num=seed)
            # env.p.stopStateLogging(logID)
            print('Evaluation: {} | Cost: {:.3f}, Goal Cost: {:.3f}, Coll Cost: {:.3f}, End Position: {:.3f}'
                  .format(j, cost, goal_cost, fall_cost, end_position))
            batch_costs[j] = torch.Tensor([cost])
        print("Average Cost for the Environment:", batch_costs.mean().item())
        env.p.disconnect()  # clean up instance
        return batch_costs

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
    num_params = sum(p.numel() for p in policy.parameters())
    print(num_params)
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
        batch_cost = test(params, policy, device, mu, std, posterior, i)
        total_cost += batch_cost.mean().item()
        print("Average Cost:", total_cost/(i-env_num_start+1))
