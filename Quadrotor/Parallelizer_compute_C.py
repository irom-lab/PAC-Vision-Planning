#!/usr/bin/env python3

import multiprocessing as mp
import numpy as np
import torch
from policy import Policy, Filter
from ES_grad import compute_grad_ES
import os
import time
import warnings
from optimize_PAC_bound import optimize_PAC_bound
warnings.filterwarnings('ignore')

class Compute_Loss:

    def __init__(self, num_trials, num_cpu=1, num_gpu=1, start_seed=0, multi_server=False):
        # self.cores = mp.cpu_count()
        self.cores = num_cpu
        self.batch = [0 for _ in range(self.cores)]  # give each core the correct number of processes + data
        for i in range(num_trials):
            self.batch[i % self.cores] += 1
        self.num_trials = num_trials
        self.num_gpu = num_gpu
        self.start_seed = start_seed
        self.multi_server = multi_server

    def compute(self, itr_num, params, mu, std, mu_pr, logvar_pr, reg_include):

        # Need this start_method for parallelizing Pytorch models
        mp.set_start_method('forkserver', force=True)
        delta = params['delta']
        process = []
        batch = self.batch
        manager = mp.Manager()
        rd = manager.dict()
        pos = self.start_seed
        torch_pos = 0
        device = [torch.device('cuda:'+str(i)) for i in range(self.num_gpu)]
        policy = Policy()
        DepthFilter = Filter()
        policy.share_memory()
        DepthFilter.share_memory()

        # Assumption: num_cpu_cores >= num_gpu
        device_list = [0] * self.cores
        for i in range(self.cores):
            # device_counter[i % self.num_gpu] += 1
            device_list[i] = i % self.num_gpu

        for j in range(self.cores):
            
            # Share the same epsilon acoss all environments for an iteration
            torch_seed = [itr_num] * batch[j]
            
            # Fixing the np_seed fixes the enviroment
            np_seed = list(range(pos,pos+batch[j]))
            process.append(mp.Process(target=self.thread, args=(params, policy, DepthFilter, device[device_list[j]],
                                                                mu, std, mu_pr, logvar_pr, batch[j],
                                                                np_seed, torch_seed, reg_include, rd, j)))
            pos += batch[j]
            torch_pos += batch[j]
            process[j].start()

        for j in range(self.cores):
            process[j].join()

        # Collect the epsilons along with cost (how to keep them separate from other environments?)
        emp_cost = []
        coll_cost = []
        goal_cost = []
        all_emp_costs = []
        for i in range(self.cores):
            # torch.cat misbehaves when there is a 0-dim tensor, hence view
            emp_cost.extend(rd['costs'+str(i)].view(1,rd['costs'+str(i)].numel()))
            coll_cost.extend(rd['coll_costs'+str(i)].view(1,rd['costs'+str(i)].numel()))
            goal_cost.extend(rd['goal_costs'+str(i)].view(1,rd['costs'+str(i)].numel()))
            all_emp_costs.extend(rd['all_emp_costs'+str(i)])
            
        emp_cost = torch.cat(emp_cost)
        goal_cost = torch.cat(goal_cost)
        coll_cost = torch.cat(coll_cost)
        
        emp_cost_stack = torch.stack(all_emp_costs)
        
        return emp_cost, coll_cost, goal_cost, emp_cost_stack

    @staticmethod
    def new_seed():
        return int(2 ** 32 * np.random.random_sample())

    @staticmethod
    def thread(params, policy, DepthFilter, device, mu, std, mu_pr, logvar_pr, batch_size, np_seed, torch_seed, reg_include, rd, proc_num):
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

        from Simulator import Simulator

        '''import pybullet results in printing "pybullet build time: XXXX" for
        each process. The code below suppresses printing these messages.
        Source: https://stackoverflow.com/a/978264'''
        # SUPPRESS PRINTING
        null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        save = os.dup(1), os.dup(2)
        os.dup2(null_fds[0], 1)
        os.dup2(null_fds[1], 2)

        from Environment import TestEnvironment

        # ENABLE PRINTING
        os.dup2(save[0], 1)
        os.dup2(save[1], 2)
        os.close(null_fds[0])
        os.close(null_fds[1])

        # creating objects
        policy_eval_costs = torch.zeros(num_policy_eval)
        batch_costs = torch.zeros(batch_size)
        coll_costs = torch.zeros(num_policy_eval)
        batch_coll_costs = torch.zeros(batch_size)
        goal_costs = torch.zeros(num_policy_eval)
        batch_goal_costs = torch.zeros(batch_size)
        all_emp_costs = torch.zeros(batch_size, num_policy_eval)

        for p in DepthFilter.parameters():
            p.data = torch.ones_like(p.data)
        DepthFilter = DepthFilter.to(device)
        
        # ====================
        # num_trials = 500
        # ====================

        env = TestEnvironment(r_lim, num_obs, safecircle=safecircle, parallel=True,
                          gui=False, y_max=y_max, y_min=y_min, x_min=x_min, x_max=x_max)
        sim = Simulator(comp_len=comp_len, prim_horizon=prim_horizon, alpha=alpha,
                        dt=time_step, t_horizon=t_horizon, device=device)  # new env for this thread to use

        # Generate epsilons in here and compute multiple runs for the same environment
        for i in range(batch_size):
            torch.manual_seed(torch_seed[i])
            epsilon = torch.randn((num_policy_eval, mu.numel()))
            # epsilon = torch.cat([epsilon, -epsilon], dim=0)
            if i>0:
                env.p.removeBody(env.obsUid)
                # Initialize the robot back to the initial point for collision check with obstacle map 
                state = env.init_position
                quat = env.init_orientation
                env.p.resetBasePositionAndOrientation(env.quadrotor, [state[0], state[1], state[2]], quat)

            np.random.seed(np_seed[i])
            env.generate_safe_initial_env(1.25)

            for j in range(num_policy_eval):
                policy_params = mu + std*epsilon[j,:]
                policy_params = policy_params.to(device)

                # LOAD POLICY_PARAMS
                count = 0
                for p in policy.parameters():
                    num_params_p = p.data.numel()
                    p.data = policy_params[count:count+num_params_p].view(p.data.shape)
                    count+=num_params_p
                cost, collision_cost, goal_cost = sim.simulate_controller(env, policy, DepthFilter,
                                                                          gen_new_env=False,
                                                                          rem_old_env=False,
                                                                          image_size=image_size)

                policy_eval_costs[j] = torch.Tensor([cost])
                coll_costs[j] = torch.Tensor([collision_cost])
                goal_costs[j] = torch.Tensor([goal_cost])

            all_emp_costs[i] = policy_eval_costs
            batch_costs[i] = policy_eval_costs.mean()
            batch_coll_costs[i] = coll_costs.mean()
            batch_goal_costs[i] = goal_costs.mean()

        # Return the sum of all costs in the batch
        rd['costs'+str(proc_num)] = batch_costs
        rd['coll_costs'+str(proc_num)] = batch_coll_costs
        rd['goal_costs'+str(proc_num)] = batch_goal_costs
        rd['all_emp_costs'+str(proc_num)] = all_emp_costs
        env.p.disconnect()  # clean up instance
