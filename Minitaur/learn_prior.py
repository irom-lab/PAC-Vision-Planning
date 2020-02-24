#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:36:39 2019

@author: sushant
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from policy import Policy
import time
import warnings
import argparse
warnings.filterwarnings('ignore')

def pr_loss(out, device):
    true = (torch.ones(11)*(1./11.)).to(device)
    return F.binary_cross_entropy(out, true, reduction='sum') / out.shape[0]

if __name__ == '__main__':

    def collect_as(coll_type):
        class Collect_as(argparse.Action):
            def __call__(self, parser, namespace, values, options_string=None):
                setattr(namespace, self.dest, coll_type(values))
        return Collect_as

    parser = argparse.ArgumentParser(description='PAC-Bayes Training')
    parser.add_argument('--num_itr', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--load', type=bool, default=False)
    parser.add_argument('--device', type=str, default="cuda")

    args = parser.parse_args()

    load = args.load
    lr = args.lr
    num_itr = args.num_itr
    batch_size = args.batch_size
    device = torch.device(args.device)

    policy = Policy()
    policy.to(device)
    num_params = sum(p.numel() for p in policy.parameters())

    optimizer = optim.Adam([ {'params': policy.parameters(), 'lr': lr}])

    for i in range(num_itr):

        optimizer.zero_grad()
        start = time.time()

        ind1 = np.random.randint(0,20000)
        ind2 = np.random.randint(0,10)
        depth = np.load('../husky-fpv-navigation/train/Dataset/Depth/depth_husky_'+str(ind1)+'_'+str(ind2)+'.npy')
        depth = torch.Tensor(depth).view(1,1,depth.shape[0],depth.shape[1]).detach().to(device)

        out = policy(depth, torch.Tensor([[0,0]]).to(device), torch.Tensor([[0,0]]).to(device))
        loss = pr_loss(out, device)
        loss.backward()

        print('Iteration: {}, time:{:.1f} s, Train Cost: {:.3f}'.format(
          i, time.time()-start, loss.item()))

        # Update the parameters
        optimizer.step()

    print(out)
    mu_pr = []
    mu_pr_dict = policy.state_dict()
    for p in list(mu_pr_dict):
        mu_pr.append(mu_pr_dict[p].view(1,-1))
    mu_pr = torch.cat(mu_pr, dim=1)[0]
    torch.save(mu_pr, 'Weights/mu_pr.pt')