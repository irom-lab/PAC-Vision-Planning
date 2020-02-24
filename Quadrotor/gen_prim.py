#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 20:18:02 2019

@author: Sushant Veer
"""

import numpy as np
import json
import sys

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def gen_traj(x_path, y_path, z_path, T, dt):
    '''
    Takes in a path, total time, and time-steps and generates a time trajectory.

    Parameters
    ----------
    x_path : x coordinates of the path
    y_path : y coordinates of the path
    z_path : z coordinates of the path
    T : total time
    dt : time steps

    Returns
    -------
    x_traj : time trajectory of the x coorindate
    y_traj : time trajectory of the y coorindate
    z_traj : time trajectory of the z coorindate
    thetax_traj : time trajectory of the angle of the total velocity with x-direction
    thetay_traj : time trajectory of the angle of the total velocity with y-direction
    thetaz_traj : time trajectory of the angle of the total velocity with z-direction
    v : constant speed of the robot along the trajectory 
    '''
    l = np.zeros(1)
    for i in range(x_path.shape[0]-1):
        l += ((x_path[i+1]-x_path[i])**2 + (y_path[i+1]-y_path[i])**2 + (z_path[i+1]-z_path[i])**2)**0.5
    v = l/T
    x_traj = []
    y_traj = []
    z_traj = []
    thetax_traj = []
    thetay_traj = []
    thetaz_traj = []
    for i in range(x_path.shape[0]-1):
        l += ((x_path[i+1]-x_path[i])**2 + (y_path[i+1]-y_path[i])**2 + (z_path[i+1]-z_path[i])**2)**0.5
        if l>=v*dt:
            x_traj.append(x_path[i])
            y_traj.append(y_path[i])
            z_traj.append(z_path[i])
            l = 0
    x_traj.append(x_path[-1])
    y_traj.append(y_path[-1])
    z_traj.append(z_path[-1])
    
    x_traj = np.array(x_traj)
    y_traj = np.array(y_traj)
    z_traj = np.array(z_traj)
    for i in range(x_traj.shape[0]-1):
        heading_direction = np.array([x_traj[i+1], y_traj[i+1], z_traj[i+1]]) - np.array([x_traj[i], y_traj[i], z_traj[i+1]])
        heading_angle_x = np.arccos(np.dot(heading_direction, [1,0,0])/(np.linalg.norm(heading_direction)))
        heading_angle_y = np.arccos(np.dot(heading_direction, [0,1,0])/(np.linalg.norm(heading_direction)))
        heading_angle_z = np.arccos(np.dot(heading_direction, [0,0,1])/(np.linalg.norm(heading_direction)))
        thetax_traj.append(heading_angle_x)
        thetay_traj.append(heading_angle_y)
        thetaz_traj.append(heading_angle_z)
    thetax_traj = np.array(thetax_traj)
    thetay_traj = np.array(thetay_traj)
    thetaz_traj = np.array(thetaz_traj)
    
    return x_traj, y_traj, z_traj, thetax_traj, thetay_traj, thetaz_traj, v

def gen_acc(v, thetax_traj, thetay_traj, thetaz_traj, dt):
    '''
    Returns the acceleration along the trajectory.
    Notation: dx, dy, dz, ddz, ddy, ddz are x_doit, y_dot, z_dot, and x_ddot,..., respectively
    Since dx = v*cos(thetax)
          ddx = -v*sin(thetax)*dthetax

    Parameters
    ----------
    v : constant speed of the robot
    thetax_traj : evolution of the angle of the robot's velocity along the global x direction
    thetay_traj : evolution of the angle of the robot's velocity along the global y direction
    thetaz_traj : evolution of the angle of the robot's velocity along the global z direction
    dt : time step

    Returns
    -------
    ddx_traj : acceleration of the trajectory along x
    ddy_traj : acceleration of the trajectory along y
    ddz_traj : acceleration of the trajectory along z

    '''
    dthetax_traj = []
    dthetay_traj = []
    dthetaz_traj = []
    for i in range(thetax_traj.shape[0]-1):
        dthetax_traj.append((thetax_traj[i+1]-thetax_traj[i])/dt)
        dthetay_traj.append((thetay_traj[i+1]-thetay_traj[i])/dt)
        dthetaz_traj.append((thetaz_traj[i+1]-thetaz_traj[i])/dt)
    dthetax_traj = np.array(dthetax_traj)
    dthetay_traj = np.array(dthetay_traj)
    dthetaz_traj = np.array(dthetaz_traj)

    ddx_traj = []
    ddy_traj = []
    ddz_traj = []
    for i in range(dthetax_traj.shape[0]):
        ddx_traj.append((-v*np.sin(thetax_traj[i])*dthetax_traj[i])[0])
        ddy_traj.append((-v*np.sin(thetay_traj[i])*dthetay_traj[i])[0])
        ddz_traj.append((-v*np.sin(thetaz_traj[i])*dthetaz_traj[i])[0])
    ddx_traj = np.array(ddx_traj)
    ddy_traj = np.array(ddy_traj)
    ddz_traj = np.array(ddz_traj)
    
    return ddx_traj, ddy_traj, ddz_traj
    

def gen_path(x0, y0, z0, dx, dz, vy, T, steps):
    '''
    Assumes a constant speed in the forward direction and moves x and z along a
    sigmoid. This is just a means to generate a path, time is essentially a proxy
    for the y coordinate.

    Parameters
    ----------
    x0 : robot initial x position
    y0 : robot initial y position
    z0 : robot initial z position
    dx : primitive displacement in x direction
    dz : primitive displacement in z direction
    vy : velocity in the forward direction
    T : total time
    steps : number of steps to discretize the path in

    Returns
    -------
    x_path: numpy array
    y_path: numpy array
    z_path: numpy array
    '''
    t = np.linspace(0, T, steps+1)
    x_traj = x0 + dx*sigmoid((t-T/2)*6)
    y_traj = y0 + t*vy
    z_traj = z0 + dz*sigmoid((t-T/2)*6)
    return x_traj[1:], y_traj[1:], z_traj[1:]
            
def gen_end_points(x_min, x_max, z_min, z_max, num_x_points=5, num_z_points=5):

    end_points = np.empty([num_x_points*num_z_points, 2], dtype=float)
    x_points = np.linspace(x_min, x_max, num_x_points)
    z_points = np.linspace(z_min, z_max, num_z_points)
    count = 0
    for i in range(len(x_points)):
        for j in range(len(z_points)):
            end_points[count,:] = [x_points[i], z_points[j]]
            count+=1
            
    # np.save('prim_lib.npy', prim_lib)
    print(end_points)
    return end_points

def gen_prim_lib(x_min, x_max, z_min, z_max, num_x_points=5, num_z_points=5, vy=1.25, dt=0.05, T=1, steps=4000):
    '''
    Generates the library of primitives.
    Ensure that vy * T = 1.25 m
    '''
    num_prims = num_x_points * num_z_points
    end_points = gen_end_points(x_min=-1, x_max=1, z_min=-1, 
                            z_max=1, num_x_points=5, num_z_points=5)
    
    prim_lib_x_traj = []
    prim_lib_y_traj = []
    prim_lib_z_traj = []
    prim_lib_x_acc = []
    prim_lib_y_acc = []
    prim_lib_z_acc = []
    prim_lib_v = []
    for i in range(num_prims):
        x_path, y_path, z_path = gen_path(x0=0, y0=0, z0=0, dx=end_points[i,0], 
                                          dz=end_points[i,1], vy=vy, T=T, steps=steps)
        x_traj, y_traj, z_traj, thetax_traj, thetay_traj, thetaz_traj, v = gen_traj(
                                                                           x_path, 
                                                                           y_path, 
                                                                           z_path, 
                                                                           T=T, 
                                                                           dt=dt)
        prim_lib_x_traj.append(x_traj)
        prim_lib_y_traj.append(y_traj)
        prim_lib_z_traj.append(z_traj)
        prim_lib_v.append(v)
        
        ddx_traj, ddy_traj, ddz_traj = gen_acc(v, thetax_traj, thetay_traj, thetaz_traj, dt=dt)
        
        prim_lib_x_acc.append(ddx_traj)
        prim_lib_y_acc.append(ddy_traj)
        prim_lib_z_acc.append(ddz_traj)
        
    prim_lib_x_traj = np.stack(prim_lib_x_traj)
    prim_lib_y_traj = np.stack(prim_lib_y_traj)
    prim_lib_z_traj = np.stack(prim_lib_z_traj)
    prim_lib_v = np.stack(prim_lib_v)
    prim_lib_x_acc = np.stack(prim_lib_x_acc)
    prim_lib_y_acc = np.stack(prim_lib_y_acc)
    prim_lib_z_acc = np.stack(prim_lib_z_acc)
    
    return prim_lib_x_traj, prim_lib_y_traj, prim_lib_z_traj, prim_lib_x_acc, prim_lib_y_acc, prim_lib_z_acc, prim_lib_v       

if __name__ == '__main__':
    
    params = {}
    params = json.load(open(sys.argv[1]))
    dt = params['time_step']
    T = params['prim_horizon'] * dt
    
    # end_points = gen_end_points(x_min=-1, x_max=1, z_min=-1, 
    #                         z_max=1, num_x_points=5, num_z_points=5)
    # x_path, y_path, z_path = gen_path(x0=0, y0=0, z0=0, dx=end_points[0,0], 
    #                                   dz=end_points[0,1], vy=1, T=0.5, steps=1000)
    # x_traj, y_traj, z_traj, thetax_traj, thetay_traj, thetaz_traj, v = gen_traj(
    #                                                                    x_path, 
    #                                                                    y_path, 
    #                                                                    z_path, 
    #                                                                    T=0.5, 
    #                                                                    dt=0.05)
    # print(x_traj)
    # print(y_traj)
    # print(z_traj)
    
    prim_lib_x_traj, prim_lib_y_traj, prim_lib_z_traj, prim_lib_x_acc, prim_lib_y_acc, prim_lib_z_acc, prim_lib_v = gen_prim_lib(
                                                x_min=-1, x_max=1, z_min=-1, z_max=1, num_x_points=5, num_z_points=5, dt=dt, T=T)
    
    print(prim_lib_x_traj[0,:])
    print(prim_lib_y_traj[0,:])
    print(prim_lib_z_traj[0,:])
    
    np.save('primitives/prim_lib_x_traj.npy', prim_lib_x_traj)
    np.save('primitives/prim_lib_y_traj.npy', prim_lib_y_traj)
    np.save('primitives/prim_lib_z_traj.npy', prim_lib_z_traj)
    np.save('primitives/prim_lib_x_acc.npy', prim_lib_x_acc)
    np.save('primitives/prim_lib_y_acc.npy', prim_lib_y_acc)
    np.save('primitives/prim_lib_z_acc.npy', prim_lib_z_acc)
    np.save('primitives/prim_lib_v.npy', prim_lib_v)

