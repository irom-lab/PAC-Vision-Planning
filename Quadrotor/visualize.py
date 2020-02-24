#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 22:20:31 2019

@author: sushant
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def cost_spread(goal_cost, coll_cost):
    # PLot the cost spread and log to tensorboard
    width = 0.25
    ind = list(range(1,goal_cost.numel()+1))
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.bar([x-width/2 for x in ind], goal_cost, width, label='Goal Cost')
    ax.bar([x+width/2 for x in ind], coll_cost, width, label='Collision Cost')
    plt.xlabel("Environment Index")
    plt.ylabel("Cost")
    plt.title('Cost Spread Over Envs')
    ax.legend()
    return fig

def cost_spread_train(cost, y_max):
    # PLot the train cost spread and log to tensorboard
    width = 0.5
    ind = list(range(1,cost.numel()+1))
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.bar([x for x in ind], cost, width, label='Collision Cost')
    plt.xlabel("Environment Index")
    plt.ylabel("Cost")
    plt.ylim((0., y_max))
    plt.xlim((0., cost.numel()+1.))
    # Turns on minor ticks without labels at a spacing of 1
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(5))
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
    
    