#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 22:20:31 2019

@author: sushant
"""

import matplotlib.pyplot as plt

def cost_spread(goal_cost, coll_cost):
    # PLot the cost spread and log to tensorboard
    width = 0.25
    ind = list(range(1,goal_cost.numel()+1))
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.bar([x-width/2 for x in ind], goal_cost, width, label='Goal Cost')
    ax.bar([x+width/2 for x in ind], coll_cost, width, label='Collision Cost')
    plt.ylim((0., 1))
    plt.xlabel("Environment Index")
    plt.ylabel("Cost")
    plt.title('Cost Spread Over Envs')
    ax.legend()
    return fig

def weight_spread(mu, std):
    num_params = mu.numel()
    ind = list(range(1,num_params+1))
    
    fig1 = plt.figure()
    plt.scatter(ind, mu, alpha=0.2)
    plt.xlabel("Parameter Index")
    plt.ylabel("Mean")
    plt.title("Mean Spread Over Parameters")
    
    fig2 = plt.figure()
    plt.scatter(ind, std, alpha=0.2)
    plt.xlabel("Parameter Index")
    plt.ylabel("Standard Deviation")
    plt.title("Standard Deviation Spread Over Parameters")
    return fig1, fig2
    
    