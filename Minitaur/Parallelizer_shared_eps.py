#!/usr/bin/env python3

import multiprocessing as mp
import numpy as np
import torch
import torch.optim as optim
from policy import Policy as Policy
import os
import warnings
warnings.filterwarnings('ignore')

class Compute_Loss:

    def __init__(self, num_trials, num_cpu=1, num_gpu=1):
        # self.cores = mp.cpu_count()
        self.cores = num_cpu
        self.batch = [0 for _ in range(self.cores)]  # give each core the correct number of processes + data
        for i in range(num_trials):
            self.batch[i % self.cores] += 1
        self.num_trials = num_trials
        self.num_gpu = num_gpu
        # self.epsilon = torch.randn((num_trials, self.mu.numel()))

    def compute(self, params, mu, std):

        # Need this start_method for parallelizing Pytorch models
        mp.set_start_method('forkserver', force=True)
        process = []
        batch = self.batch
        manager = mp.Manager()
        rd = manager.dict()
        pos = 0
        device = [torch.device('cuda:'+str(i)) for i in range(self.num_gpu)]
        policy = Policy()
        policy.share_memory()

        # Assumption: num_cpu_cores >= num_gpu
        core_per_gpu = int(self.cores/self.num_gpu)
        device_counter = -1

        # Share the same torch seed with each thread to generate the same set of epsilons
        torch_seed = self.new_seed()

        for j in range(self.cores):
            if j % core_per_gpu == 0:
                device_counter += 1
            # Fixing the np_seed fixes the enviroment
            np_seed = list(range(pos,pos+batch[j]))
            process.append(mp.Process(target=self.thread, args=(params, policy, device[device_counter], mu, std,
                                                                batch[j], pos, np_seed, torch_seed, rd, j)))
            pos += batch[j]
            process[j].start()

        for j in range(self.cores):
            process[j].join()

        grad_method = params['grad_method']
        num_fit_frac = params['num_fit']
        num_policy_eval = params['num_policy_eval']

        # Collect the epsilons along with costs
        grad_mu = torch.zeros(mu.numel())
        grad_logvar = torch.zeros(std.numel())
        emp_cost_list = torch.zeros(2*num_policy_eval)
        emp_cost = torch.zeros(1)
        coll_cost = torch.zeros(1)
        goal_cost = torch.zeros(1)
        for i in range(self.cores):
            emp_cost += rd['costs'+str(i)]
            coll_cost += rd['coll_costs'+str(i)]
            goal_cost += rd['goal_costs'+str(i)]
            emp_cost_list += rd['cost_list'+str(i)]
        emp_cost /= self.num_trials
        coll_cost /= self.num_trials
        goal_cost /= self.num_trials
        emp_cost_list /= self.num_trials

        torch.manual_seed(torch_seed)

        epsilon = torch.randn((num_policy_eval, mu.numel()))
        epsilon = torch.cat([epsilon, -epsilon], dim=0)

        grad_mu, grad_logvar = self.compute_grad(emp_cost_list, epsilon, std, grad_method, num_fit_frac=num_fit_frac)

        return emp_cost, grad_mu, grad_logvar, coll_cost, goal_cost

    @staticmethod
    def new_seed():
        return int(2 ** 32 * np.random.random_sample())

    @staticmethod
    def compute_grad(policy_eval_costs, epsilons, std, method='fitness_sampling', unpert_policy_eval_cost=torch.zeros(1), num_fit_frac=1):
        num_policy_eval = int(policy_eval_costs.shape[0]/2)

        if method=='fitness_sampling':
            '''Scale by utility function u instead of the loss'''

            # Fitness rank transformation
            u = torch.zeros(num_policy_eval*2)
            for i in range(num_policy_eval*2):
                u[i] = torch.max(torch.Tensor([torch.Tensor([num_policy_eval+1]).log() - torch.Tensor([i+1]).log(), 0]))
            u /= u.sum()
            u -= 1./(2*num_policy_eval)

            fit_index = policy_eval_costs.sort().indices
            epsilons = epsilons[fit_index]
    
            loss = u
            grad_mu = (torch.matmul(torch.transpose(epsilons,0,1), loss) / std)/epsilons.shape[0]
            grad_std = (torch.matmul(torch.transpose(epsilons**2-torch.ones_like(epsilons),0,1), loss) / std)/epsilons.shape[0]
            logvar = torch.log(std.pow(2))
            grad_logvar = 0.5 * grad_std * (logvar*0.5).exp()
            # The direction with the lowest cost is given the highest utility,
            # so effectively we are finding the negative of the gradient
            return -grad_mu, -grad_logvar
    
        if method=='ES':
            '''Pick num_fit best epsilon and scale by 1-loss function'''
            fit_index = policy_eval_costs.sort().indices
            num_fit = int(2*num_policy_eval*num_fit_frac)
            fit_index = fit_index[:num_fit]
    
            if num_fit_frac < 1:
                loss = 1. - policy_eval_costs[fit_index]
            else:
                loss = policy_eval_costs[fit_index]
            epsilons = epsilons[fit_index]
            grad_mu = (torch.matmul(torch.transpose(epsilons,0,1), loss) / std)/epsilons.shape[0]
            grad_std = (torch.matmul(torch.transpose(epsilons**2-torch.ones_like(epsilons),0,1), loss) / std)/epsilons.shape[0]
            logvar = torch.log(std.pow(2))
            grad_logvar = 0.5 * grad_std * (logvar*0.5).exp()
            if num_fit_frac < 1:
                grad_mu *= -1
                grad_logvar *= -1
            return grad_mu, grad_logvar
    
        if method=='RBO':
            '''See equation (5) in https://arxiv.org/abs/1903.02993'''
            y = policy_eval_costs - unpert_policy_eval_cost
            Z = epsilons*std
            v = torch.randn_like(std, requires_grad=True)
            optimizer = optim.Adam([v], 1e-1)
            for i in range(100):
                optimizer.zero_grad()
                loss = ((y-torch.matmul(Z,v))/(4*num_policy_eval)).norm()**2 + 0.1 * v.norm()**2
                loss.backward()
                optimizer.step()
    
            grad_mu = v.detach()
            grad_logvar = torch.zeros_like(grad_mu)
            return grad_mu, grad_logvar
    
        if method=='mirror_ES':
            '''Pick num_fit best (epsilon, -epsilon) pairs to ensure termination
            grad is achieved'''
            policy_eval_costs_pos_eps = policy_eval_costs[:num_policy_eval]
            policy_eval_costs_neg_eps = policy_eval_costs[num_policy_eval:]
            policy_eval_costs_aug = torch.zeros([2,num_policy_eval])
            policy_eval_costs_aug[0,:] = policy_eval_costs_pos_eps
            policy_eval_costs_aug[1,:] = policy_eval_costs_neg_eps
            policy_eval_costs_min = torch.min(policy_eval_costs_aug, dim=0).values
            fit_index = policy_eval_costs_min.sort().indices
            num_fit = int(num_policy_eval*num_fit_frac)
            fit_index = fit_index[:num_fit]
            loss = 1 - torch.cat([policy_eval_costs_pos_eps[fit_index], policy_eval_costs_neg_eps[fit_index]], dim=0)
            epsilons = torch.cat([epsilons[fit_index], -epsilons[fit_index]], dim=0)
    
            grad_mu = (torch.matmul(torch.transpose(epsilons,0,1), loss) / std)/epsilons.shape[0]
            grad_std = (torch.matmul(torch.transpose(epsilons**2-torch.ones_like(epsilons),0,1), loss) / std)/epsilons.shape[0]
            logvar = torch.log(std.pow(2))
            grad_logvar = 0.5 * grad_std * (logvar*0.5).exp()
            return -grad_mu, -grad_logvar

    @staticmethod
    def thread(params, policy, device, mu, std, batch_size, ind_start, np_seed, torch_seed, rd, proc_num):
        time_step = params['time_step']
        husky_velocity = params['husky_velocity']
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
        from Sensor import DepthSensor

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

        from Robot import Robot

        # creating objects
        sim = Simulator(comp_len=comp_len, prim_horizon=prim_horizon, alpha=alpha,
                        t_horizon=t_horizon, device=device)  # new env for this thread to use
        robot = Robot(DepthSensor(), forward_speed=husky_velocity, dt=time_step)
        env = TestEnvironment(robot, r_lim, num_obs, safecircle=safecircle, parallel=True,
                              gui=False, y_max=y_max, y_min=y_min, x_min=x_min, x_max=x_max)
        policy_eval_costs = torch.zeros([batch_size, num_policy_eval*2])
        batch_costs = torch.zeros(batch_size)
        coll_costs = torch.zeros(num_policy_eval*2)
        batch_coll_costs = torch.zeros(batch_size)
        goal_costs = torch.zeros(num_policy_eval*2)
        batch_goal_costs = torch.zeros(batch_size)

        # Generate epsilons in here and compute multiple runs for the same environment
        for i in range(batch_size):
            np.random.seed(seed=np_seed[i])
            torch.manual_seed(torch_seed)
            epsilon = torch.randn((num_policy_eval, mu.numel()))
            epsilon = torch.cat([epsilon, -epsilon], dim=0)
            if i>0:
                env.obsUid = env.generate_obstacles()

            for j in range(num_policy_eval*2+1):
                if j == num_policy_eval*2:
                    policy_params = mu
                else:
                    policy_params = mu + std*epsilon[j,:]

                policy_params = policy_params.to(device)

                # LOAD POLICY_PARAMS
                count = 0
                for p in policy.parameters():
                    num_params_p = p.data.numel()
                    p.data = policy_params[count:count+num_params_p].view(p.data.shape)
                    count+=num_params_p

                cost, collision_cost, goal_cost, _, _, _ = sim.simulate_controller(env, policy,
                                                                                   gen_new_env=False,
                                                                                   rem_old_env=False,
                                                                                   image_size=image_size)
                # print(torch.Tensor([cost]))
                if j == num_policy_eval*2:
                    unpert_policy_eval_cost = torch.Tensor([cost])
                else:
                    policy_eval_costs[i,j] = torch.Tensor([cost])
                    coll_costs[j] = torch.Tensor([collision_cost])
                    goal_costs[j] = torch.Tensor([goal_cost])

            batch_costs[i] = policy_eval_costs[i,:].mean()
            batch_coll_costs[i] = coll_costs.mean()
            batch_goal_costs[i] = goal_costs.mean()

        # Return the entire list of costs
        rd['cost_list'+str(proc_num)] = policy_eval_costs.sum(dim=0)

        # Return the sum of all costs in the batch
        rd['costs'+str(proc_num)] = batch_costs.sum()
        rd['coll_costs'+str(proc_num)] = batch_coll_costs.sum()
        rd['goal_costs'+str(proc_num)] = batch_goal_costs.sum()
        env.p.disconnect()  # clean up instance