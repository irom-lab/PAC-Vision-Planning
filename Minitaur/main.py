#!/usr/bin/env python3

from Environment import Environment
from Parallelizer import Compute_Loss
import warnings
import time
import numpy as np
import torch
from policy import Policy
warnings.filterwarnings('ignore')

# Start time counter
start = time.time()

params = {}
params['num_trials'] = 10  # number of clips you want for training purposes
params['video_length'] = 2  # number of images per video clip
params['folder'] = '../train/Dataset/data_files/'  # data location
# Creates a directory if one does not exist
# if not os.path.isdir(params['folder']):
#             os.mkdir(params['folder'])

q = np.zeros(50)
# q += 0.02  # random controller
q[12] = 1  # pretty good controller
# q[25] = 1  # good controller
params['q'] = q  # Controller distribution
params['data_filter'] = True  # try to get good data, start w/ 0 cost, end clip with 0 or 1 etc.

# PARAMETERS
params['time_step'] = 0.05  # seconds
params['husky_velocity'] = 5  # meters per second
params['image_size'] = 50  # square
# ground_covered_per_clip = video_length * time_step * husky_velocity

# Time horizon of a single primitive
params['prim_horizon'] = 4
# Number of primitive compositions for the policy
params['comp_len'] = 10
# Time horizon of an entire run
params['t_horizon'] = params['prim_horizon'] * params['comp_len']
params['goal_thresh'] = 0.5

# OBSTACLE PARAMETERS
params['r_lim'] = [0.3, 0.5]  # obstacle radius options
params['num_obs'] = 5  # number of obstacles

# ENVIRONMENT INFORMATION
params['y_max']=15
params['y_min']=-5
params['x_min']=-10
params['x_max']=10

policy = Policy()
num_params = sum(p.numel() for p in policy.parameters())
mu = torch.randn(num_params)
std = torch.randn(num_params)

start = time.time()
para = Compute_Loss(params['num_trials'], mu, std)
costs, epsilons = para.compute(params)
end = time.time()
print("Computation Time:", end-start)

def compute_grad(mu, std, epsilons, loss):
    grad_mu = (torch.matmul(torch.transpose(epsilons,0,1), loss) / std)/epsilons.shape[0]
    grad_std = (torch.matmul(torch.transpose(epsilons**2-torch.ones_like(epsilons),0,1), loss) / std)/epsilons.shape[0]
    return grad_mu, grad_std

grad_mu, grad_std = compute_grad(mu, std, epsilons, costs)
print(grad_mu)
print(grad_std)