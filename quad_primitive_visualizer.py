#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 09:26:03 2019

@author: sushant
"""

import torch
import sys
import json
import warnings
from envs.Quad_Simulator import Simulator
from envs.Quad_Env import Environment
import numpy as np
import time
warnings.filterwarnings('ignore')

def plot_primitive_traj(params, device):
        time_step = params['time_step']
        t_horizon = params['t_horizon']
        r_lim = params['r_lim']
        num_obs = 0
        y_max=params['y_max']
        y_min=params['y_min']
        x_min=params['x_min']
        x_max=params['x_max']
        comp_len = params['comp_len']
        prim_horizon = params['prim_horizon']
        num_policy_eval = params['num_policy_eval']
        alpha = params['alpha']

        # creating objects
        env = Environment(r_lim, num_obs, parallel=True,
                          gui=True, y_max=y_max, y_min=y_min, x_min=x_min, x_max=x_max)
        sim = Simulator(comp_len=comp_len, prim_horizon=prim_horizon, alpha=alpha,
                        dt=time_step, t_horizon=t_horizon, device=device)
        batch_costs = torch.zeros(num_policy_eval)

        np.random.seed(0)
        if num_obs > 0:
            env.obsUid = env.generate_obstacles()

        collision_flag = 0
        collision_counter = 0
        
        max_angle_across_prims = 0
        
        for j in range(25):
            angle, rgb, depth = sim.visualize_prims(j, env, gen_new_env=False, rem_old_env=False, image_size=50)
            if angle > max_angle_across_prims:
                max_angle_across_prims = angle
                
        print("Minimum FOV required: "+str(2*max_angle_across_prims)+" degrees")
            
        time.sleep(60)
        env.p.disconnect()  # clean up instance
        return batch_costs, collision_flag, collision_counter

if __name__ == "__main__":

    params = {}
    params = json.load(open(sys.argv[1]))
    params['num_policy_eval'] = 25
    save_file_v = params['save_file_v']
    
    device = torch.device("cuda")
    plot_primitive_traj(params, device)