#!/usr/bin/env python3

import multiprocessing as mp
import numpy as np
import torch
from policy import Policy as Policy
import os
from ES_grad import compute_grad_ES
import warnings
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
        process = []
        batch = self.batch
        manager = mp.Manager()
        rd = manager.dict()
        pos = self.start_seed
        torch_pos = 0
        device = [torch.device('cuda:'+str(i)) for i in range(self.num_gpu)]
        policy = Policy()
        policy.share_memory()

        # Assumption: num_cpu_cores >= num_gpu
        device_list = [0] * self.cores
        for i in range(self.cores):
            # device_counter[i % self.num_gpu] += 1
            device_list[i] = i % self.num_gpu

        for j in range(self.cores):
            # Generate seeds at random with no regard to repeatability
            # torch_seed = [self.new_seed() for i in range(batch[j])]
            
            # Share the same epsilon acoss all environments for an iteration
            # torch_seed = [itr_num] * batch[j]

            # Don't share epsilons among environment
            # Note: Do not use this with multiserver, otherwise epsilons will repeat
            #       across servers
            torch_seed = list(range(itr_num*self.num_trials + torch_pos, 
                                    itr_num*self.num_trials + torch_pos + batch[j]))
            
            # Fixing the np_seed fixes the enviroment
            np_seed = list(range(pos,pos+batch[j]))
            process.append(mp.Process(target=self.thread, args=(params, policy, device[device_list[j]],
                                                                mu, std, mu_pr, logvar_pr, batch[j],
                                                                np_seed, torch_seed, reg_include, rd, j)))
            pos += batch[j]
            torch_pos += batch[j]
            process[j].start()

        for j in range(self.cores):
            process[j].join()

        # Collect the epsilons along with cost (how to keep them separate from other environments?)
        grad_mu = torch.zeros(mu.numel())
        grad_logvar = torch.zeros(std.numel())
        emp_cost = []
        coll_cost = []
        goal_cost = []

        for i in range(self.cores):
            grad_mu += rd[i][0]
            grad_logvar += rd[i][1]

            # torch.cat misbehaves when there is a 0-dim tensor, hence view
            emp_cost.extend(rd['costs'+str(i)].view(1,rd['costs'+str(i)].numel()))
            coll_cost.extend(rd['coll_costs'+str(i)].view(1,rd['costs'+str(i)].numel()))
            goal_cost.extend(rd['goal_costs'+str(i)].view(1,rd['costs'+str(i)].numel()))

        emp_cost = torch.cat(emp_cost)
        goal_cost = torch.cat(goal_cost)
        coll_cost = torch.cat(coll_cost)

        # If computing across servers, only pass the cumulative sum, normalization
        # will take place in the head_node
        if self.multi_server:
            return emp_cost, grad_mu, grad_logvar, coll_cost, goal_cost
        else:
            grad_mu /= self.num_trials
            grad_logvar /= self.num_trials

        return emp_cost.detach(), grad_mu.detach(), grad_logvar.detach(), coll_cost.detach(), goal_cost.detach()

    @staticmethod
    def new_seed():
        return int(2 ** 32 * np.random.random_sample())

    @staticmethod
    def thread(params, policy, device, mu, std, mu_pr, logvar_pr, batch_size, np_seed, torch_seed, reg_include, rd, proc_num):
        image_size = params['image_size']
        max_angle = params['max_angle']
        comp_len = params['comp_len']
        prim_horizon = params['prim_horizon']
        num_policy_eval = params['num_policy_eval']
        alpha = params['alpha']
        grad_method = params['grad_method']
        num_fit_frac = params['num_fit']
        time_step = params['time_step']
        goal = params['goal']

        '''import pybullet results in printing "pybullet build time: XXXX" for
        each process. The code below suppresses printing these messages.
        Source: https://stackoverflow.com/a/978264'''
        # SUPPRESS PRINTING
        null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        save = os.dup(1), os.dup(2)
        os.dup2(null_fds[0], 1)
        os.dup2(null_fds[1], 2)

        from Environment import Environment

        # ENABLE PRINTING
        os.dup2(save[0], 1)
        os.dup2(save[1], 2)
        os.close(null_fds[0])
        os.close(null_fds[1])

        # creating objects
        policy_eval_costs = torch.zeros(num_policy_eval*2)
        grad_mu = torch.zeros(mu.numel())
        grad_logvar = torch.zeros(std.numel())
        batch_costs = torch.zeros(batch_size)
        coll_costs = torch.zeros(num_policy_eval*2)
        batch_coll_costs = torch.zeros(batch_size)
        goal_costs = torch.zeros(num_policy_eval*2)
        batch_goal_costs = torch.zeros(batch_size)

        env = Environment(max_angle=max_angle, gui=False)

        # Generate epsilons in here and compute multiple runs for the same environment
        for i in range(batch_size):
            torch.manual_seed(torch_seed[i])
            epsilon = torch.randn((num_policy_eval, mu.numel()))
            epsilon = torch.cat([epsilon, -epsilon], dim=0)
            if i>0:
                env.p.removeBody(env.terrain)
            env.goal = goal
            np.random.seed(np_seed[i])
            env.terrain = env.generate_steps()

            for j in range(num_policy_eval*2):
                # reset the robot back to starting point after each trial
                env.minitaur_env.reset()
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

                cost, collision_cost, goal_cost, _ = env.execute_policy(policy,
                                                                     env.goal,
                                                                     alpha,
                                                                     time_step=time_step,
                                                                     comp_len=comp_len,
                                                                     prim_horizon=prim_horizon,
                                                                     image_size=image_size,
                                                                     device=device)

                policy_eval_costs[j] = torch.Tensor([cost])
                coll_costs[j] = torch.Tensor([collision_cost])
                goal_costs[j] = torch.Tensor([goal_cost])

            batch_costs[i] = policy_eval_costs.mean()
            batch_coll_costs[i] = coll_costs.mean()
            batch_goal_costs[i] = goal_costs.mean()

            grad_mu_temp, grad_logvar_temp = compute_grad_ES(policy_eval_costs-policy_eval_costs.mean(), 
                                                             epsilon, 
                                                             std, 
                                                             num_fit_frac, 
                                                             grad_method, 
                                                             0)

            grad_mu += grad_mu_temp
            grad_logvar += grad_logvar_temp

        # Gradient is computed for 1-loss, so return its negation as the true gradient
        rd[proc_num] = [grad_mu, grad_logvar]

        # Return the sum of all costs in the batch
        rd['costs'+str(proc_num)] = batch_costs
        rd['coll_costs'+str(proc_num)] = batch_coll_costs
        rd['goal_costs'+str(proc_num)] = batch_goal_costs
        env.p.disconnect()  # clean up instance