#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 18:29:04 2019

@author: sushant
"""


import os
import json
import sys

args={}
args = json.load(open(sys.argv[1]))
server_list = args['server_list']
for i in range(len(server_list)):
    os.system('server_utils/shut_down_server.sh '+server_list[i])
    # os.system('server_utils/sync_server.sh '+server_list[i])
    
