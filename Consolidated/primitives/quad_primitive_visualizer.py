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
from Simulator import Simulator
from Environment import TestEnvironment
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import cv2
warnings.filterwarnings('ignore')

def plot_primitive_traj(params, device):
        time_step = params['time_step']
        image_size = params['image_size']
        t_horizon = params['t_horizon']
        r_lim = params['r_lim']
        num_obs = 23
        y_max=params['y_max']
        y_min=params['y_min']
        x_min=params['x_min']
        x_max=params['x_max']
        comp_len = params['comp_len']
        prim_horizon = params['prim_horizon']
        num_policy_eval = params['num_policy_eval']
        safecircle = params['safecircle']
        alpha = params['alpha']

        # creating objects
        sim = Simulator(comp_len=comp_len, prim_horizon=prim_horizon, alpha=alpha,
                        dt=time_step, t_horizon=t_horizon, device=device)  # new env for this thread to use
        env = TestEnvironment(r_lim, num_obs, safecircle=safecircle, parallel=False, gui=True,
                          y_max=y_max, y_min=y_min, x_min=x_min, x_max=x_max)
        batch_costs = torch.zeros(num_policy_eval)

        np.random.seed(1)
        if num_obs > 0:
            env.obsUid = env.generate_obstacles()

        collision_flag = 0
        collision_counter = 0
        
        max_angle_across_prims = 0
        
        for j in range(25):
            angle, rgb, depth = sim.visualize_prims(j, env, gen_new_env=False, rem_old_env=False, image_size=50)
            if angle > max_angle_across_prims:
                max_angle_across_prims = angle
                
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        # Turns on minor ticks without labels at a spacing of 1
        # ax.xaxis.set_minor_locator(MultipleLocator(1))
        # ax.xaxis.set_major_locator(MultipleLocator(5))
        # ax.yaxis.set_minor_locator(MultipleLocator(1))
        # ax.yaxis.set_major_locator(MultipleLocator(5))

        # rgb = rgb[9:34, 13:38, :]
        plt.imshow(rgb, cmap='gray', interpolation='nearest')
        plt.savefig('quadrotor_prims.png')

        depth.astype('uint16')
        depth = cv2.convertScaleAbs(depth, alpha=255./depth.max())
        plt.imshow(depth, cmap='gray')#, interpolation='nearest')
        plt.savefig('depth_map.png')

        print("Minimum FOV required: "+str(2*max_angle_across_prims)+" degrees")
            
        time.sleep(600)
        env.p.disconnect()  # clean up instance
        return batch_costs, collision_flag, collision_counter

if __name__ == "__main__":

    params = {}
    params = json.load(open(sys.argv[1]))
    params['num_policy_eval'] = 25
    save_file_v = params['save_file_v']
    
    device = torch.device("cuda")
    plot_primitive_traj(params, device)