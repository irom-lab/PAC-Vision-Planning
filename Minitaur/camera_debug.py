#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:20:57 2020

@author: sushant
"""

import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data
import numpy as np

p = bc.BulletClient(connection_mode=pybullet.DIRECT)

view_matrix = (0.0025001317262649536, -0.3049148917198181, -0.9523764252662659, 0.0, 
               -0.7796670198440552, -0.5969661474227905, 0.18907921016216278, 0.0, 
               -0.6261894702911377, 0.7420637607574463, -0.23922453820705414, 0.0, 
               0.04080526530742645, -0.38639146089553833, 0.18126174807548523, 1.0)

proj_matrix = (0.9999999403953552, 0.0, 0.0, 0.0, 
               0.0, 0.9999999403953552, 0.0, 0.0, 
               0.0, 0.0, -1.0000200271606445, -1.0, 
               0.0, 0.0, -0.02000020071864128, 0.0)

_, _, rgb, depth, _ = p.getCameraImage(50, 50, view_matrix, proj_matrix, flags=p.ER_NO_SEGMENTATION_MASK)

print(np.array(depth).shape)