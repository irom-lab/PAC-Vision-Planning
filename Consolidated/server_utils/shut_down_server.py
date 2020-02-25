#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 21:40:14 2019

@author: sushant
"""


import os
import json
import sys

args={}
args = json.load(open(sys.argv[1]))
server_list = args['server_list']
for i in range(len(server_list)):
    os.system('./server_utils/shut_down_server.sh '+server_list[i])
    