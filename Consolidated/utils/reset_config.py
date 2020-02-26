#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 13:38:48 2019

@author: sushant
"""
import json
import os
import sys

if len(sys.argv)>2:
    delete = sys.argv[2]
else:
    delete = '0'

args={}
args = json.load(open(sys.argv[1]))

args['itr_start'] = 0
args['load_prior'] = False
args['load_weights'] = False
args['load_optimizer'] = False
args['cost_min'] = 100.0

json.dump(args, open(sys.argv[1], 'w'), indent=4)

if delete == '1':
    os.system('rm -rf runs/summary_'+args['save_file_v'])
