#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 20:18:02 2019

@author: Sushant Veer
"""

import numpy as np

def gen_prim(num_prims=5, prim_horizon=50):

    prim_lib = np.linspace(0.5, 1., num=num_prims)
    print(prim_lib)
    np.save('prim_lib.npy', prim_lib)

if __name__ == '__main__':
    num_prims = 5
    prim_horizon = 4
    gen_prim(num_prims, prim_horizon)
